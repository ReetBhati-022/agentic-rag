/* ===========================================================================
   Multi-Format RAG Assistant — Frontend Application Logic
=========================================================================== */
(function () {
  "use strict";

  const body = document.getElementById("app");

  // -------------------------------------------------------------------
  // Small helpers
  // -------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  function showToast(message, isError) {
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "toast" + (isError ? " error" : " success");
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
    setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // Persistent "in progress" toast with a spinner. Used for uploads started
  // from the workspace (+ Add document / chat attach), because the landing
  // page's progress overlay is hidden there. showToast() removes it, so the
  // success/error toast simply replaces it.
  function showBusyToast(message) {
    document.querySelectorAll(".toast").forEach((t) => t.remove());
    const toast = document.createElement("div");
    toast.className = "toast busy";
    const spinner = document.createElement("span");
    spinner.className = "toast-spinner";
    const text = document.createElement("span");
    text.textContent = message;
    toast.append(spinner, text);
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
  }

  function setView(hasIndex) {
    body.classList.toggle("state-workspace", hasIndex);
    body.classList.toggle("state-landing", !hasIndex);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // -------------------------------------------------------------------
  // Minimal Markdown -> HTML renderer for assistant answers.
  //
  // The backend prompt now asks the LLM for plain conversational text
  // that may use light Markdown (bold, bullet/numbered lists, small
  // tables, the occasional heading) -- this turns that into real HTML
  // instead of showing raw "**bold**"/"- item" text to the user. It is
  // intentionally small (no external dependency, matches the "no
  // React/Vue, keep it Flask + vanilla JS" constraint) and only covers
  // the handful of constructs a RAG answer realistically uses. All text
  // is escaped before any HTML is introduced, so this is safe against
  // an answer that happens to contain "<" or "&".
  // -------------------------------------------------------------------
  function inlineMarkdown(text) {
    let out = escapeHtml(text);
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    return out;
  }

  function splitTableRow(row) {
    return row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((s) => s.trim());
  }

  function renderMarkdown(src) {
    if (!src) return "";
    const lines = src.replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let listType = null; // "ul" | "ol" | null
    let paragraph = [];
    let tableLines = [];

    function closeList() {
      if (listType) { out.push(`</${listType}>`); listType = null; }
    }
    function flushParagraph() {
      if (paragraph.length) {
        out.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
        paragraph = [];
      }
    }
    function flushTable() {
      if (!tableLines.length) return;
      const isTable = tableLines.length >= 2 && /^[\s|:-]+$/.test(tableLines[1]);
      if (!isTable) {
        tableLines.forEach((l) => paragraph.push(l));
        tableLines = [];
        return;
      }
      const header = splitTableRow(tableLines[0]);
      out.push("<table><thead><tr>" + header.map((h) => `<th>${inlineMarkdown(h)}</th>`).join("") + "</tr></thead><tbody>");
      tableLines.slice(2).forEach((row) => {
        const cells = splitTableRow(row);
        out.push("<tr>" + cells.map((c) => `<td>${inlineMarkdown(c)}</td>`).join("") + "</tr>");
      });
      out.push("</tbody></table>");
      tableLines = [];
    }

    for (const raw of lines) {
      const line = raw.trim();

      if (/^\|.*\|$/.test(line)) {
        flushParagraph();
        closeList();
        tableLines.push(line);
        continue;
      }
      if (tableLines.length) flushTable();

      if (!line) {
        flushParagraph();
        closeList();
        continue;
      }

      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        flushParagraph();
        closeList();
        const level = Math.min(heading[1].length + 3, 6);
        out.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }

      const bullet = line.match(/^[-*•]\s+(.*)$/);
      if (bullet) {
        flushParagraph();
        if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; }
        out.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
        continue;
      }

      const numbered = line.match(/^\d+[.)]\s+(.*)$/);
      if (numbered) {
        flushParagraph();
        if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; }
        out.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
        continue;
      }

      closeList();
      paragraph.push(line);
    }
    flushParagraph();
    closeList();
    if (tableLines.length) flushTable();

    return out.join("\n");
  }

  // =====================================================================
  // 1. UPLOAD / INDEXING FLOW (landing view)
  // =====================================================================
  const dropzone = $("dropzone");
  const fileInput = $("fileInput");
  const fileStage = $("fileStage");
  const fileStageList = $("fileStageList");
  const processBtn = $("processBtn");

  let stagedFiles = [];
  let appendMode = false; // becomes true when uploading via "+ Add document"

  function refreshFileStage() {
    if (stagedFiles.length === 0) {
      fileStage.hidden = true;
      return;
    }
    fileStage.hidden = false;
    fileStageList.innerHTML = "";
    stagedFiles.forEach((file, idx) => {
      const pill = document.createElement("div");
      pill.className = "file-pill";
      pill.innerHTML =
        `<span>${escapeHtml(file.name)}</span>` +
        `<button type="button" class="file-remove" data-idx="${idx}" aria-label="Remove file">&times;</button>`;
      fileStageList.appendChild(pill);
    });
  }

  function addFiles(fileList) {
    const incoming = Array.from(fileList || []);
    const allowed = [".pdf", ".png", ".jpg", ".jpeg"];
    const valid = incoming.filter((f) => {
      const ext = "." + f.name.split(".").pop().toLowerCase();
      return allowed.includes(ext);
    });
    if (valid.length < incoming.length) {
      showToast("Some files were skipped — only PDF, PNG, and JPG are supported.", true);
    }
    stagedFiles = stagedFiles.concat(valid);
    refreshFileStage();
  }

  if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      addFiles(e.dataTransfer.files);
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      addFiles(fileInput.files);
      fileInput.value = "";
    });
  }

  if (fileStageList) {
    fileStageList.addEventListener("click", (e) => {
      const btn = e.target.closest(".file-remove");
      if (!btn) return;
      const idx = Number(btn.dataset.idx);
      stagedFiles.splice(idx, 1);
      refreshFileStage();
    });
  }

  // ---- Progress overlay -------------------------------------------------
  const progressOverlay = $("progressOverlay");
  const progressStages = $("progressStages") ? Array.from($("progressStages").children) : [];
  const progressError = $("progressError");

  const STAGE_SEQUENCE = ["upload", "extract", "ocr", "chunk", "embed", "faiss", "ready"];
  // Approximate relative durations so the bar doesn't feel linear/fake.
  const STAGE_TIMING_MS = [400, 1400, 1800, 1000, 1600, 900, 300];

  let progressTimer = null;

  function resetProgressUI() {
    progressStages.forEach((li) => {
      li.classList.remove("active", "done", "error");
      const bar = li.querySelector(".stage-bar i");
      if (bar) bar.style.width = "0%";
    });
    progressError.hidden = true;
    progressError.textContent = "";
  }

  function openProgressOverlay() {
    resetProgressUI();
    fileStage.hidden = true;
    progressOverlay.hidden = false;
    runSimulatedStages();
  }

  function closeProgressOverlay() {
    if (progressTimer) clearTimeout(progressTimer);
    progressOverlay.hidden = true;
  }

  // The backend runs ingestion synchronously and doesn't stream stage
  // events, so we animate through the known pipeline stages while the
  // single upload request is in flight, then snap everything to "done"
  // as soon as the response actually arrives (see uploadFiles()).
  function runSimulatedStages() {
    let i = 0;
    function step() {
      if (i > 0) {
        const prevLi = progressStages[i - 1];
        prevLi.classList.remove("active");
        prevLi.classList.add("done");
        prevLi.querySelector(".stage-bar i").style.width = "100%";
      }
      if (i >= STAGE_SEQUENCE.length - 1) return; // leave "ready" for completion
      const li = progressStages[i];
      li.classList.add("active");
      const bar = li.querySelector(".stage-bar i");
      requestAnimationFrame(() => { bar.style.width = "85%"; });
      progressTimer = setTimeout(step, STAGE_TIMING_MS[i]);
      i++;
    }
    step();
  }

  function finishProgressUI(success, errorMessage) {
    if (progressTimer) clearTimeout(progressTimer);
    if (success) {
      progressStages.forEach((li) => {
        li.classList.remove("active", "error");
        li.classList.add("done");
        li.querySelector(".stage-bar i").style.width = "100%";
      });
    } else {
      progressStages.forEach((li) => {
        if (li.classList.contains("done")) return;
        li.classList.remove("active");
        li.classList.add("error");
      });
      progressError.hidden = false;
      progressError.textContent = errorMessage || "Something went wrong while indexing.";
    }
  }

  let isUploading = false;

  function setUploadBusy(busy) {
    const addBtn = $("addDocBtn");
    if (addBtn) {
      addBtn.disabled = busy;
      addBtn.textContent = busy ? "Adding…" : "+ Add document";
    }
    const attachBtn = $("chatAttachBtn");
    if (attachBtn) attachBtn.disabled = busy;
  }

  async function uploadFiles() {
    if (stagedFiles.length === 0) return;
    if (isUploading) {
      showToast("An upload is already in progress — please wait.", true);
      return;
    }
    isUploading = true;

    // In the workspace the progress overlay is hidden, so surface progress
    // through a toast instead.
    const inWorkspace = body.classList.contains("state-workspace");
    const uploadLabel = stagedFiles.length === 1
      ? stagedFiles[0].name
      : `${stagedFiles.length} files`;
    if (inWorkspace) {
      setUploadBusy(true);
      showBusyToast(`Adding ${uploadLabel} — extracting, chunking and indexing…`);
    }

    const form = new FormData();
    stagedFiles.forEach((f) => form.append("files", f));
    form.append("append", appendMode ? "true" : "false");

    openProgressOverlay();

    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        finishProgressUI(false, data.error || "Indexing failed.");
        if (inWorkspace) showToast(data.error || "Indexing failed.", true);
        setTimeout(() => {
          closeProgressOverlay();
          refreshFileStage();
        }, 2600);
        return;
      }

      finishProgressUI(true);
      setTimeout(() => {
        closeProgressOverlay();
        stagedFiles = [];
        refreshFileStage();
        appendMode = false;
        setView(true);
        applyIndexedState(data);
        // Rebuild the preview selector from the server's authoritative
        // list (covers both fresh-upload replacement and "+ Add
        // document" appending), then load whichever document is now
        // active.
        populatePdfDocSelect(data.pdf_documents || [], data.active_pdf_name);
        loadPdfViewer(currentPdfName);
      }, 650);
    } catch (err) {
      finishProgressUI(false, "Network error — is the server running?");
      if (inWorkspace) showToast("Network error — is the server running?", true);
      setTimeout(() => {
        closeProgressOverlay();
        refreshFileStage();
      }, 2600);
    } finally {
      isUploading = false;
      if (inWorkspace) setUploadBusy(false);
    }
  }

  if (processBtn) processBtn.addEventListener("click", uploadFiles);

  function applyIndexedState(data) {
    $("workspace-topbar-actions").hidden = false;

    const pipelineSummary = $("pipelineSummary");
    const pipelineSteps = $("pipelineSteps");
    if (pipelineSummary && pipelineSteps) {
      const steps = ["Text Extraction", "Chunking", "Embeddings", "FAISS Indexing"];
      pipelineSteps.innerHTML = steps
        .map((label) => `<span class="pipeline-step"><span class="pipeline-step-check">&#10003;</span>${escapeHtml(label)}</span>`)
        .join("");
      pipelineSummary.hidden = false;
    }

    const docStats = $("docStats");
    if (docStats) {
      docStats.innerHTML = "";
      const totalDocs = (data.pdf_documents || []).length || data.num_docs;
      const stats = [
        { label: "Documents", value: totalDocs ?? "—" },
        { label: "Chunks", value: data.total_chunks ?? data.num_chunks ?? "—" },
        { label: "Embedding Model", value: (data.embedding_model || "").split("/").pop() || "—" },
      ];
      stats.forEach((s) => {
        const el = document.createElement("div");
        el.className = "doc-stat";
        el.innerHTML = `<span>${escapeHtml(s.label)}</span>${escapeHtml(String(s.value))}`;
        docStats.appendChild(el);
      });
    }

    const fileLabel = (data.files && data.files.length === 1)
      ? data.files[0]
      : `${(data.files || []).length} files`;
    const totalCount = (data.pdf_documents || []).length || data.num_docs;
    showToast(`✓ ${fileLabel} added successfully — ${totalCount} document(s) indexed in total.`, false);
  }

  // =====================================================================
  // 2. WORKSPACE TOP BAR — provider picker, add doc, clear index
  // =====================================================================
  const providerBtn = $("providerBtn");
  const providerMenu = $("providerMenu");
  const providerLabel = $("providerLabel");
  let currentProvider = body.dataset.provider || "ollama";

  if (providerBtn) {
    providerBtn.addEventListener("click", () => {
      const open = !providerMenu.hidden;
      providerMenu.hidden = open;
      providerBtn.setAttribute("aria-expanded", String(!open));
    });
    document.addEventListener("click", (e) => {
      if (!providerBtn.contains(e.target) && !providerMenu.contains(e.target)) {
        providerMenu.hidden = true;
        providerBtn.setAttribute("aria-expanded", "false");
      }
    });
    providerMenu.addEventListener("click", (e) => {
      const opt = e.target.closest(".provider-option");
      if (!opt) return;
      currentProvider = opt.dataset.provider;
      providerLabel.textContent = currentProvider.charAt(0).toUpperCase() + currentProvider.slice(1);
      providerMenu.hidden = true;
    });
  }

  const addDocBtn = $("addDocBtn");
  if (addDocBtn) {
    addDocBtn.addEventListener("click", () => {
      appendMode = true;
      const tempInput = document.createElement("input");
      tempInput.type = "file";
      tempInput.multiple = true;
      tempInput.accept = ".pdf,.png,.jpg,.jpeg";
      tempInput.addEventListener("change", () => {
        stagedFiles = Array.from(tempInput.files);
        uploadFiles();
      });
      tempInput.click();
    });
  }

  const clearIndexBtn = $("clearIndexBtn");
  if (clearIndexBtn) {
    clearIndexBtn.addEventListener("click", async () => {
      if (!confirm("This will permanently delete the current index. Continue?")) return;
      try {
        const res = await fetch("/api/clear", { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          showToast(data.error || "Failed to clear index.", true);
          return;
        }
        setView(false);
        $("chatMessages").innerHTML = "";
        pdfDoc = null;
        populatePdfDocSelect([], null);
        updateDownloadLink(null);
        renderEmptyPdfState();
      } catch (err) {
        showToast("Network error while clearing the index.", true);
      }
    });
  }

  // =====================================================================
  // 3. PDF VIEWER (PDF.js)
  // =====================================================================
  const pdfCanvas = $("pdfCanvas");
  const pdfCanvasWrap = $("pdfCanvasWrap");
  const pdfEmpty = $("pdfEmpty");
  const pdfThumbnails = $("pdfThumbnails");
  const pdfPageInput = $("pdfPageInput");
  const pdfPageCount = $("pdfPageCount");
  const pdfZoomLabel = $("pdfZoomLabel");
  const pdfDownload = $("pdfDownload");
  const pdfDocPicker = $("pdfDocPicker");
  const pdfDocSelect = $("pdfDocSelect");

  let pdfDoc = null;
  let currentPage = 1;
  let currentZoom = 1.0;
  let currentRotation = 0;
  let renderTask = null;

  // ---- Multi-document preview state --------------------------------
  // View-only: which of the currently-indexed PDFs the viewer is
  // showing right now, and the full list to populate the selector
  // with. This never touches chat/retrieval state -- it only decides
  // what URL loadPdfViewer() fetches from /pdf/<name>.
  let pdfDocuments = [];
  let currentPdfName = null;

  if (window.pdfjsLib) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.min.js";
  }

  function renderEmptyPdfState() {
    pdfEmpty.hidden = false;
    pdfCanvas.hidden = true;
  }

  // Rebuilds the document selector from the given list of indexed PDF
  // filenames. Keeps `currentPdfName` in sync with whatever ends up
  // selected (the requested active name if it's still in the list,
  // otherwise the first document, otherwise none).
  function populatePdfDocSelect(list, preferredName) {
    pdfDocuments = Array.isArray(list) ? list.slice() : [];

    if (!pdfDocSelect || !pdfDocPicker) return;
    pdfDocSelect.innerHTML = "";

    if (pdfDocuments.length === 0) {
      pdfDocPicker.hidden = true;
      currentPdfName = null;
      return;
    }

    pdfDocuments.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      pdfDocSelect.appendChild(opt);
    });

    const chosen = (preferredName && pdfDocuments.includes(preferredName))
      ? preferredName
      : pdfDocuments[0];
    pdfDocSelect.value = chosen;
    currentPdfName = chosen;
    pdfDocPicker.hidden = false;
  }

  function buildPdfPreviewUrl(name) {
    // Falls back to the legacy /pdf/current endpoint if no specific
    // document name is known yet (keeps single-PDF/backward-compat
    // callers working unchanged).
    return name
      ? "/pdf/" + encodeURIComponent(name) + "?_=" + Date.now()
      : "/pdf/current?_=" + Date.now();
  }

  function updateDownloadLink(name) {
    if (!pdfDownload) return;
    pdfDownload.href = name ? "/pdf/" + encodeURIComponent(name) : "/pdf/current";
  }

  // loadPdfViewer(name) loads a specific indexed PDF into the existing
  // PDF.js viewer, resetting page/zoom/rotation and rebuilding
  // thumbnails -- reused for both the initial load and every selector
  // change, rather than duplicated per call site. `name` is optional;
  // omitted it falls back to the current selection (or /pdf/current).
  async function loadPdfViewer(name) {
    if (!window.pdfjsLib) {
      renderEmptyPdfState();
      pdfEmpty.querySelector("p").textContent =
        "PDF viewer couldn't load (pdf.js failed to load from the CDN — check your network/firewall).";
      showToast("PDF preview unavailable — pdf.js failed to load. Chat still works normally.", true);
      return;
    }
    const targetName = name || currentPdfName;
    try {
      const loadingTask = pdfjsLib.getDocument(buildPdfPreviewUrl(targetName));
      pdfDoc = await loadingTask.promise;
      pdfEmpty.hidden = true;
      pdfCanvas.hidden = false;
      currentPage = 1;
      currentZoom = 1.0;
      currentRotation = 0;
      pdfZoomLabel.textContent = "100%";
      pdfPageCount.textContent = pdfDoc.numPages;
      pdfPageInput.max = pdfDoc.numPages;
      await renderPage(currentPage);
      buildThumbnails();

      currentPdfName = targetName || currentPdfName;
      if (pdfDocSelect && currentPdfName) pdfDocSelect.value = currentPdfName;
      updateDownloadLink(currentPdfName);
    } catch (err) {
      renderEmptyPdfState();
      pdfEmpty.querySelector("p").textContent =
        "Couldn't load the PDF for preview (see console for details). Chat still works normally.";
      console.error("PDF viewer load failed:", err);
      showToast("Couldn't load the PDF preview — check the console for details.", true);
    }
  }

  if (pdfDocSelect) {
    pdfDocSelect.addEventListener("change", () => {
      const name = pdfDocSelect.value;
      if (!name || name === currentPdfName) return;
      // Switching the preview only ever reloads the PDF.js viewer --
      // no re-upload, no re-index, chat/retrieval state is untouched.
      loadPdfViewer(name);
    });
  }

  async function renderPage(num) {
    if (!pdfDoc) return;
    if (renderTask) {
      try { renderTask.cancel(); } catch (e) { /* ignore */ }
    }
    const page = await pdfDoc.getPage(num);
    const viewport = page.getViewport({ scale: currentZoom * 1.4, rotation: currentRotation });
    pdfCanvas.width = viewport.width;
    pdfCanvas.height = viewport.height;
    const ctx = pdfCanvas.getContext("2d");
    renderTask = page.render({ canvasContext: ctx, viewport });
    try {
      await renderTask.promise;
    } catch (e) { /* cancelled render, ignore */ }
    currentPage = num;
    pdfPageInput.value = num;
    document.querySelectorAll(".pdf-thumb").forEach((el) => {
      el.classList.toggle("active", Number(el.dataset.page) === num);
    });
  }

  async function buildThumbnails() {
    pdfThumbnails.innerHTML = "";
    if (!pdfDoc) return;
    const maxThumbs = Math.min(pdfDoc.numPages, 30); // sane cap for very long docs
    for (let n = 1; n <= maxThumbs; n++) {
      const page = await pdfDoc.getPage(n);
      const viewport = page.getViewport({ scale: 0.18 });
      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;

      const wrap = document.createElement("div");
      wrap.className = "pdf-thumb" + (n === currentPage ? " active" : "");
      wrap.dataset.page = n;
      wrap.appendChild(canvas);
      const label = document.createElement("div");
      label.className = "pdf-thumb-num";
      label.textContent = n;
      wrap.appendChild(label);
      wrap.addEventListener("click", () => renderPage(n));
      pdfThumbnails.appendChild(wrap);
    }
  }

  $("pdfPrev").addEventListener("click", () => { if (pdfDoc && currentPage > 1) renderPage(currentPage - 1); });
  $("pdfNext").addEventListener("click", () => { if (pdfDoc && currentPage < pdfDoc.numPages) renderPage(currentPage + 1); });
  pdfPageInput.addEventListener("change", () => {
    if (!pdfDoc) return;
    let n = Number(pdfPageInput.value);
    n = Math.max(1, Math.min(pdfDoc.numPages, n || 1));
    renderPage(n);
  });
  $("pdfZoomIn").addEventListener("click", () => {
    if (!pdfDoc) return;
    currentZoom = Math.min(3, currentZoom + 0.15);
    pdfZoomLabel.textContent = Math.round(currentZoom * 100) + "%";
    renderPage(currentPage);
  });
  $("pdfZoomOut").addEventListener("click", () => {
    if (!pdfDoc) return;
    currentZoom = Math.max(0.4, currentZoom - 0.15);
    pdfZoomLabel.textContent = Math.round(currentZoom * 100) + "%";
    renderPage(currentPage);
  });
  $("pdfRotate").addEventListener("click", () => {
    if (!pdfDoc) return;
    currentRotation = (currentRotation + 90) % 360;
    renderPage(currentPage);
  });

  // Citations can point at any indexed document, not just the one
  // currently shown (multiple PDFs may be indexed at once). If the
  // citation names a different, still-previewable document, switch
  // the viewer to it first -- otherwise this would silently render
  // the right page number in the wrong PDF. `sourceName` is optional;
  // omitted, this behaves exactly as before (jump within the current
  // document only).
  function jumpToPage(pageNum, sourceName) {
    const scrollAndClose = () => pdfCanvasWrap.scrollIntoView({ behavior: "smooth", block: "start" });
    if (sourceName && pdfDocuments.includes(sourceName) && sourceName !== currentPdfName) {
      loadPdfViewer(sourceName).then(() => {
        if (!pdfDoc) return;
        const n = Math.max(1, Math.min(pdfDoc.numPages, Number(pageNum) || 1));
        renderPage(n);
        scrollAndClose();
      });
      return;
    }
    if (!pdfDoc) return;
    const n = Math.max(1, Math.min(pdfDoc.numPages, Number(pageNum) || 1));
    renderPage(n);
    scrollAndClose();
  }

  // =====================================================================
  // 4. CHAT
  // =====================================================================
  const chatMessages = $("chatMessages");
  const chatForm = $("chatForm");
  const chatInput = $("chatInput");
  const chatSendBtn = $("chatSendBtn");
  const chatWelcome = $("chatWelcome");

  function autoResize() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(120, chatInput.scrollHeight) + "px";
  }
  chatInput.addEventListener("input", autoResize);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  function appendUserMessage(text) {
    if (chatWelcome) chatWelcome.remove();
    const div = document.createElement("div");
    div.className = "msg msg-user";
    div.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendPendingAssistant() {
    const div = document.createElement("div");
    div.className = "msg msg-assistant pending";
    div.innerHTML = `<div class="msg-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
  }

  // Citations arrive already deduplicated by (source, page) from the
  // backend (see PromptBuilder._build_citations). The UI keeps them
  // compact by default -- a small "Sources · N" row of clickable chips
  // -- and only reveals the full evidence snippets (the old large cards)
  // if the user opts in via "Show supporting passages", instead of
  // dumping every retrieved chunk under the answer.
  function renderCitations(citations) {
    if (!citations || citations.length === 0) return "";

    // Document citations and web citations answer different questions --
    // a document chunk may only have been retrieved to resolve *what*
    // "this bank"/"this company" refers to, while the actual claims in
    // the answer come from web_search. Lumping both into one flat
    // "Sources" list makes the PDF look like evidence for web-sourced
    // claims it never supported. Split them into two labeled groups
    // instead so each citation's role is clear at a glance.
    const docCitations = citations.filter((c) => c.type !== "web");
    const webCitations = citations.filter((c) => c.type === "web");

    function docChip(c) {
      return `
        <button type="button" class="source-chip" data-page="${escapeHtml(String(c.page))}" data-source="${escapeHtml(c.source)}">
          ${escapeHtml(c.source)} · p.${escapeHtml(String(c.page))}
        </button>
      `;
    }
    function webChip(c) {
      const outlet = c.known_outlet ? " &#9733;" : "";
      return `
        <button type="button" class="source-chip source-chip-web" data-url="${escapeHtml(c.url || "")}">
          ${escapeHtml(c.source)}${outlet} &#8599;
        </button>
      `;
    }
    function docCard(c) {
      return `
        <button type="button" class="citation-card" data-page="${escapeHtml(String(c.page))}" data-source="${escapeHtml(c.source)}">
          <div class="citation-head">
            <span class="citation-source">${escapeHtml(c.source)}</span>
            <span class="citation-badge">Page ${escapeHtml(String(c.page))}</span>
          </div>
          ${c.snippet ? `<p class="citation-snippet">${escapeHtml(c.snippet)}</p>` : ""}
          <span class="citation-link">View in PDF &rarr;</span>
        </button>
      `;
    }
    function webCard(c) {
      const date = c.published_date ? escapeHtml(c.published_date) : "date not reported";
      const badge = c.known_outlet ? "Known outlet" : "Unfamiliar outlet";
      return `
        <button type="button" class="citation-card" data-url="${escapeHtml(c.url || "")}">
          <div class="citation-head">
            <span class="citation-source">${escapeHtml(c.source)}</span>
            <span class="citation-badge">${badge}</span>
          </div>
          <p class="citation-snippet">${date}</p>
          <span class="citation-link">Open source &rarr;</span>
        </button>
      `;
    }

    const groups = [];
    if (docCitations.length) {
      groups.push({
        label: docCitations.length === 1 ? "From your document" : `From your document(s) · ${docCitations.length}`,
        chips: docCitations.map(docChip).join(""),
        cards: docCitations.map(docCard).join(""),
      });
    }
    if (webCitations.length) {
      groups.push({
        label: webCitations.length === 1 ? "From the web" : `From the web · ${webCitations.length}`,
        chips: webCitations.map(webChip).join(""),
        cards: webCitations.map(webCard).join(""),
      });
    }

    const groupsHtml = groups.map((g) => `
      <div class="sources-group">
        <div class="sources-label">${g.label}</div>
        <div class="sources-compact">${g.chips}</div>
        <div class="sources-detail">${g.cards}</div>
      </div>
    `).join("");

    return `
      <div class="sources-block">
        ${groupsHtml}
        <button type="button" class="sources-toggle" aria-expanded="false">
          <span class="sources-chevron">&#9656;</span>
          <span class="sources-toggle-label">Show supporting passages</span>
        </button>
      </div>
    `;
  }

  function renderFollowups(questions) {
    if (!questions || questions.length === 0) return "";
    const chips = questions
      .map((q) => `<button type="button" class="chip followup-chip" data-q="${escapeHtml(q)}">${escapeHtml(q)}</button>`)
      .join("");
    return `
      <div class="followup-block">
        <div class="sources-label">You might also ask</div>
        <div class="followup-chips">${chips}</div>
      </div>
    `;
  }

  function replacePendingWithAnswer(pendingEl, answer, citations, warnings, suggestedQuestions) {
    pendingEl.classList.remove("pending");
    const warningHtml = warnings && warnings.length
      ? `<p class="msg-warning">${escapeHtml(warnings[0])}</p>`
      : "";
    pendingEl.innerHTML =
      `<div class="msg-bubble">${renderMarkdown(answer)}</div>` +
      renderCitations(citations) +
      warningHtml +
      renderFollowups(suggestedQuestions);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function sendQuestion(question) {
    if (!question || !question.trim()) return;
    appendUserMessage(question);
    const pendingEl = appendPendingAssistant();
    chatSendBtn.disabled = true;

    try {
      // Routed through the agent graph (plan -> tools -> synthesize ->
      // evaluate) instead of plain RAG, so document retrieval, web
      // search (Exa), and the math tool are all considered per query
      // -- matches the flow: query -> agent -> decide tools (PDF /
      // web / math) -> synthesize -> human-readable answer.
      const res = await fetch("/api/agent-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, provider: currentProvider }),
      });
      const data = await res.json();

      if (!res.ok) {
        pendingEl.classList.remove("pending");
        pendingEl.innerHTML = `<div class="msg-bubble">${escapeHtml(data.error || "Something went wrong.")}</div>`;
        return;
      }
      replacePendingWithAnswer(pendingEl, data.answer, data.citations, data.warnings, data.suggested_questions);
    } catch (err) {
      pendingEl.classList.remove("pending");
      pendingEl.innerHTML = `<div class="msg-bubble">Network error — please try again.</div>`;
    } finally {
      chatSendBtn.disabled = false;
    }
  }

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = chatInput.value;
    chatInput.value = "";
    autoResize();
    sendQuestion(q);
  });

  chatMessages.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) {
      sendQuestion(chip.dataset.q);
      return;
    }

    // Compact source chip or an expanded evidence card -- a document
    // citation jumps the PDF viewer to that page; a web citation opens
    // the source URL in a new tab instead.
    const sourceChip = e.target.closest(".source-chip");
    if (sourceChip) {
      if (sourceChip.dataset.url) {
        window.open(sourceChip.dataset.url, "_blank", "noopener");
      } else {
        jumpToPage(sourceChip.dataset.page, sourceChip.dataset.source);
      }
      return;
    }
    const citation = e.target.closest(".citation-card");
    if (citation) {
      if (citation.dataset.url) {
        window.open(citation.dataset.url, "_blank", "noopener");
      } else {
        jumpToPage(citation.dataset.page, citation.dataset.source);
      }
      return;
    }

    // "Show/Hide supporting passages" -- reveals the detailed evidence
    // snippets for this answer only; collapsed by default so the chat
    // doesn't dump every retrieved chunk under every message.
    const toggle = e.target.closest(".sources-toggle");
    if (toggle) {
      const block = toggle.closest(".sources-block");
      const isOpen = block.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(isOpen));
      toggle.querySelector(".sources-toggle-label").textContent =
        isOpen ? "Hide supporting passages" : "Show supporting passages";
    }
  });

  const clearChatBtn = $("clearChatBtn");
  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/clear-chat", { method: "POST" });
      } catch (e) { /* best-effort */ }
      chatMessages.innerHTML = "";
    });
  }

  const chatAttachBtn = $("chatAttachBtn");
  const chatAttachInput = $("chatAttachInput");
  if (chatAttachBtn && chatAttachInput) {
    chatAttachBtn.addEventListener("click", () => chatAttachInput.click());
    chatAttachInput.addEventListener("change", async () => {
      if (!chatAttachInput.files.length) return;
      appendMode = true;
      stagedFiles = Array.from(chatAttachInput.files);
      chatAttachInput.value = "";
      await uploadFiles();
    });
  }

  // =====================================================================
  // 5. INIT — reflects actual server state, never assumes an in-progress
  //    upload. The progress overlay stays untouched (hidden) unless
  //    uploadFiles() is actually running.
  // =====================================================================
  document.addEventListener("DOMContentLoaded", () => {
    const hasIndex = body.dataset.indexReady === "true";
    setView(hasIndex);
    if (hasIndex) {
      $("workspace-topbar-actions").hidden = false;

      let initialDocs = [];
      const pdfDocumentsDataEl = $("pdfDocumentsData");
      if (pdfDocumentsDataEl) {
        try {
          initialDocs = JSON.parse(pdfDocumentsDataEl.textContent || "[]");
        } catch (e) { initialDocs = []; }
      }
      populatePdfDocSelect(initialDocs, body.dataset.activePdf);

      if (currentPdfName) {
        loadPdfViewer(currentPdfName);
      } else if (body.dataset.activePdf) {
        // Fallback for the (rare) case the selector list is empty but
        // the server still has a legacy active PDF -- keeps
        // backward-compat with /pdf/current.
        loadPdfViewer();
      } else {
        renderEmptyPdfState();
      }
    }
  });
})();