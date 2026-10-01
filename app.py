from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from flasgger import Swagger
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from src.Ingestion.llmwhisperer_extractor import LLMWhispererExtractionError, LLMWhispererExtractor
from src.chunking.chunker import Chunker
from src.llm.llm_factory import LLMFactory, SUPPORTED_PROVIDERS
from src.retrieval.faiss_store import FaissStore
from src.retrieval.relevance import check_relevance
from src.retrieval.retriever import Retriever
from src.services.extraction_store import (
    clear_extraction_text,
    save_extraction_document,
)
from src.services.chunking_store import clear_chunking_text, save_chunking_text
from src.services.faiss_log import clear_faiss_log, save_faiss_log
from src.services.memory_store import MemoryStore
from src.agent.graph import run_agent
from src.agent.nodes import generate_followup_questions

# ---------------------------------------------------------------------------
# Logging -- WITHOUT this, Python's root logger defaults to WARNING, which
# silently drops every logger.info(...) call in the codebase. This makes
# them visible on stdout, at the level Flask/werkzeug's own request-log
# lines already print at.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# App / config setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Loads .env (EXA_API_KEY, GROQ_API_KEY, OLLAMA_HOST, etc.) into the real
# process environment. python-dotenv was already a requirements.txt
# dependency but was never actually invoked anywhere -- os.environ.get()
# calls elsewhere (llm_factory.py, web_search.py) only ever saw real
# OS/conda environment variables without this. Must run before anything
# below reads an env var.
load_dotenv(ROOT / ".env")

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

app = Flask(
    __name__,
    template_folder=str(ROOT / "Ui-ux" / "templates"),
    static_folder=str(ROOT / "Ui-ux" / "static"),
)

# Swagger UI -> http://<host>:<port>/apidocs/  (raw spec: /apispec_1.json)
# OpenAPI 3 is required so Swagger UI shows a proper multi-file picker.
app.config["SWAGGER"] = {
    "title": CONFIG["app"].get("name", "RAG Assistant") + " API",
    "uiversion": 3,
    "openapi": "3.0.2",
    "specs_route": "/apidocs/",
}
Swagger(
    app,
    template={
        "info": {
            "title": CONFIG["app"].get("name", "RAG Assistant") + " API",
            "version": str(CONFIG["app"].get("version", "1.0.0")),
        }
    },
)

_max_upload_mb = float(CONFIG["app"].get("max_upload_size_mb", 200))
app.config["MAX_CONTENT_LENGTH"] = int(_max_upload_mb * 1024 * 1024)

UPLOAD_DIR = Path(CONFIG["app"].get("upload_dir", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = set(CONFIG["app"].get("allowed_extensions", [".pdf", ".png", ".jpg", ".jpeg"]))

# ---------------------------------------------------------------------------
# Singletons: LLMWhispererExtractor, Chunker, FaissStore, LLMFactory
# (heavy / stateful objects, built once per process).
# ---------------------------------------------------------------------------
_llmwhisperer_cfg = CONFIG.get("ingestion", {}).get("llmwhisperer", {})
_extractor = LLMWhispererExtractor(
    mode=_llmwhisperer_cfg.get("mode", "form"),
    output_mode=_llmwhisperer_cfg.get("output_mode", "layout_preserving"),
    wait_timeout=_llmwhisperer_cfg.get("wait_timeout", 300),
)
_chunker = Chunker(CONFIG)
_store = FaissStore(CONFIG)
_store.load()  # restore a previously saved index, if any, on boot
_llm_factory = LLMFactory(CONFIG["llm"])
_retriever = Retriever(_store, CONFIG)
# Long-term (cross-session) memory -- see src/services/memory_store.py.
# Reuses the same embedding model FaissStore already loaded for
# retrieval, so this doesn't pull in a second model. Connects lazily on
# first real use, same as the graph.py checkpointer.
_memory_store = MemoryStore(CONFIG, embeddings=_store.embeddings)

# ---------------------------------------------------------------------------
# Active-document persistence
# ---------------------------------------------------------------------------
# `_store` (FAISS index + chunks) already persists to disk and survives a
# restart. STATE below does NOT -- it's a plain in-memory dict, so
# active_pdf_name/active_pdf_path/indexed_files were previously lost on
# every restart even though the underlying index (and therefore chat/
# retrieval) still worked. Two small, additive pieces of recovery:
#
# 1. `indexed_files` is reconstructed from the persisted chunks'
#    `.source` field (no new storage needed -- it's already in
#    faiss_index.chunks.pkl via FaissStore.load()).
# 2. `active_pdf_path`/`active_pdf_name` (needed by /pdf/current to
#    serve actual PDF bytes, which chunk text alone can't provide) are
#    persisted in a tiny sidecar JSON next to the FAISS index. Uploaded
#    PDFs already live in a real (non-temp-cleaned) directory under
#    UPLOAD_DIR -- see api_upload below -- so the path in the sidecar
#    stays valid across restarts unless that directory was manually
#    removed, which is checked for below.
_ACTIVE_PDF_SIDECAR = Path(CONFIG["app"].get("vectorstore_dir", "data/vectorstore")) / "active_pdf.json"


def _save_active_pdf_metadata(name: str, path: str, pdf_paths: dict) -> None:
    _ACTIVE_PDF_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    with open(_ACTIVE_PDF_SIDECAR, "w", encoding="utf-8") as f:
        json.dump(
            {"active_pdf_name": name, "active_pdf_path": path, "pdf_paths": pdf_paths},
            f,
        )


def _clear_active_pdf_metadata() -> None:
    if _ACTIVE_PDF_SIDECAR.exists():
        _ACTIVE_PDF_SIDECAR.unlink()


def _load_active_pdf_metadata() -> tuple[Optional[str], Optional[str], dict]:
    """Returns (active_pdf_name, active_pdf_path, pdf_paths), or
    (None, None, {}) if no sidecar exists. `pdf_paths` is filtered down to
    entries whose file still exists on disk (the index/chat still works
    either way -- this only gates which PDFs the viewer can open)."""
    if not _ACTIVE_PDF_SIDECAR.exists():
        return None, None, {}
    try:
        with open(_ACTIVE_PDF_SIDECAR, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None, {}
    name, path = data.get("active_pdf_name"), data.get("active_pdf_path")
    pdf_paths = {
        n: p for n, p in (data.get("pdf_paths") or {}).items() if p and os.path.exists(p)
    }
    if not path or not os.path.exists(path):
        name, path = None, None
    return name, path, pdf_paths


def _reconstruct_indexed_files() -> list[str]:
    """Rebuilds the indexed_files list from persisted chunk metadata
    (each Chunk's `.source`), preserving first-seen order, deduped."""
    seen = []
    for chunk in _store.chunks:
        if chunk.source not in seen:
            seen.append(chunk.source)
    return seen


STATE = {
    "index_ready": _store.is_ready,
    "indexed_files": [],
    "chat_history": [],
    "llm_provider": CONFIG["llm"].get("default_provider", "ollama"),
    "active_pdf_path": None,
    "active_pdf_name": None,
    # Maps every indexed PDF's display name -> its on-disk path, so the
    # viewer can open any indexed document via /pdf/<name>, not just the
    # first/active one. See _save_active_pdf_metadata / api_upload.
    "pdf_paths": {},
    # LangGraph checkpoint thread id for the agentic route (see
    # src/agent/graph.py:run_agent). Regenerated wherever chat_history is
    # reset (api_clear, api_clear-chat) so a "new chat" also starts the
    # agent graph from a clean persisted state rather than resuming a
    # stale thread. Not persisted across process restarts by design --
    # on restart this becomes a new thread id, so a prior process's
    # checkpointed rows in Postgres are simply orphaned (harmless; they
    # just sit unused) rather than silently resumed into.
    "thread_id": uuid.uuid4().hex,
    # Stable long-term-memory identity. Deliberately NOT regenerated
    # anywhere thread_id is (api_clear, api_clear-chat) -- long-term
    # memory's whole point is to survive "New chat"/index resets; only a
    # real multi-user login would need this to vary. Single fixed value
    # since this app has no auth/multi-user concept.
    "user_id": CONFIG.get("agent", {}).get("memory", {}).get("default_user_id", "local_user"),
}

if _store.is_ready:
    STATE["indexed_files"] = _reconstruct_indexed_files()
    _restored_name, _restored_path, _restored_pdf_paths = _load_active_pdf_metadata()
    STATE["pdf_paths"] = _restored_pdf_paths
    if _restored_path:
        STATE["active_pdf_name"] = _restored_name
        STATE["active_pdf_path"] = _restored_path
    # else: index/chat/citations still work fine; the PDF viewer just
    # stays in its empty state (no file was found to serve) -- handled
    # already by the existing "no active_pdf_name" template/JS path.

STATE_LOCK = threading.Lock()


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _remove_dirs(dirs: list[str]) -> None:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def _error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _state_snapshot_for_template() -> dict:
    with STATE_LOCK:
        return {
            "index_ready": STATE["index_ready"],
            "indexed_files": list(STATE["indexed_files"]),
            "pdf_documents": list(STATE["indexed_files"]),
            "chat_history": list(STATE["chat_history"]),
            "llm_provider": STATE["llm_provider"],
            "active_pdf_name": STATE["active_pdf_name"],
        }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        **_state_snapshot_for_template(),
        providers=SUPPORTED_PROVIDERS,
    )


# ---------------------------------------------------------------------------
# API — upload & index
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload and index one or more documents
    Extracts text from every file, chunks it, embeds it and stores it in FAISS.
    Send each file under the same form field name `files` (multiple parts).
    Allowed types: .pdf, .png, .jpg, .jpeg (see `allowed_extensions` in config.yaml).
    ---
    tags:
      - Ingestion
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            required:
              - files
            properties:
              files:
                type: array
                description: One or more documents to index.
                items:
                  type: string
                  format: binary
              append:
                type: boolean
                default: false
                description: >
                  true = add to the existing index,
                  false = clear the index and start fresh with this batch.
    responses:
      200:
        description: All files were extracted, chunked and indexed.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                num_docs:
                  type: integer
                num_chunks:
                  type: integer
                  description: Chunks created from this upload batch.
                total_chunks:
                  type: integer
                  description: Total chunks in the index after this upload.
                files:
                  type: array
                  items:
                    type: string
                pdf_documents:
                  type: array
                  items:
                    type: string
                active_pdf_name:
                  type: string
                  nullable: true
                embedding_model:
                  type: string
      400:
        description: No file sent, unsupported file type or invalid filename.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
      500:
        description: Unexpected error during extraction or indexing.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
    """
    files = request.files.getlist("files")
    append_mode = request.form.get("append", "false").lower() == "true"

    if not files or all(f.filename == "" for f in files):
        return _error_response("Please choose at least one file to upload.")

    for f in files:
        if not _allowed_file(f.filename):
            return _error_response(
                f"Unsupported file type: '{f.filename}'. Allowed: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

    tmp_dirs: list[str] = []  # removed again if this upload fails
    try:
        # Stage 1: extract + chunk EVERY file before touching the current index, so a
        # failed/timed-out extraction can't wipe an index that was working.
        staged = []  # (safe_name, normalized_document, chunks)
        new_pdf_paths = {}
        first_pdf_path = None
        first_pdf_name = None

        for f in files:
            safe_name = secure_filename(f.filename)
            if not safe_name:
                _remove_dirs(tmp_dirs)
                return _error_response(f"Invalid filename: '{f.filename}'.")

            tmp_dir = tempfile.mkdtemp(dir=str(UPLOAD_DIR))
            tmp_dirs.append(tmp_dir)
            tmp_path = os.path.join(tmp_dir, safe_name)
            f.save(tmp_path)

            normalized_document = _extractor.extract(tmp_path, safe_name)
            chunks = _chunker.chunk(normalized_document)
            staged.append((safe_name, normalized_document, chunks))

            if safe_name.lower().endswith(".pdf"):
                new_pdf_paths[safe_name] = tmp_path
                if first_pdf_path is None:
                    first_pdf_path = tmp_path
                    first_pdf_name = safe_name

        # Stage 2: everything extracted fine -> now replace/extend the index.
        if not append_mode:
            _store.clear()
            clear_extraction_text()
            clear_chunking_text()
            clear_faiss_log()

        # Debug previews are written before embedding, so they exist even if embedding fails.
        save_extraction_document(
            document_type=staged[0][1].document_type,
            parser_used=staged[0][1].parser_used,
            content="\n\n".join(f"===== {name} =====\n{doc.full_text}" for name, doc, _ in staged),
        )
        save_chunking_text([c for _, _, cs in staged for c in cs])

        total_chunks = 0
        new_names = []
        for safe_name, _doc, chunks in staged:
            _store.add_chunks(chunks)
            total_chunks += len(chunks)
            new_names.append(safe_name)

        # FaissStore owns the actual FAISS persistence (the binary
        # data/vectorstore/*.faiss / *.chunks.pkl files); faiss.txt is
        # just a readable summary of that index, regenerated afterward.
        _store.save()
        save_faiss_log(_store)

        with STATE_LOCK:
            STATE["index_ready"] = True
            if append_mode:
                STATE["indexed_files"].extend(new_names)
                STATE["pdf_paths"].update(new_pdf_paths)
            else:
                STATE["indexed_files"] = new_names
                STATE["pdf_paths"] = new_pdf_paths
                STATE["chat_history"] = []
            if first_pdf_path:
                STATE["active_pdf_path"] = first_pdf_path
                STATE["active_pdf_name"] = first_pdf_name
            if STATE["active_pdf_path"]:
                _save_active_pdf_metadata(
                    STATE["active_pdf_name"], STATE["active_pdf_path"], STATE["pdf_paths"]
                )
            active_pdf_name = STATE["active_pdf_name"]

        return jsonify(
            success=True,
            num_docs=len(new_names),
            num_chunks=total_chunks,
            total_chunks=len(_store.chunks),
            files=new_names,
            pdf_documents=list(STATE["indexed_files"]),
            active_pdf_name=active_pdf_name,
            embedding_model=CONFIG["embeddings"]["model_name"],
        )
    except LLMWhispererExtractionError as e:
        _remove_dirs(tmp_dirs)
        print(f"[LLMWhisperer extraction error] {e}")
        return _error_response(str(e))
    except Exception as e:  # noqa: BLE001
        _remove_dirs(tmp_dirs)
        traceback.print_exc()
        return _error_response(f"Unexpected error: {e}", 500)


# ---------------------------------------------------------------------------
# API — chat (plain RAG, no agent loop)
# ---------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Ask a question (plain RAG)
    Retrieves relevant chunks, checks relevance, and answers from them. No agent loop.
    ---
    tags:
      - Chat
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - question
            properties:
              question:
                type: string
                example: What is the closing balance?
              provider:
                type: string
                enum: [ollama, groq, huggingface]
                description: LLM provider. Defaults to the server's current provider.
    responses:
      200:
        description: Answer with citations.
        content:
          application/json:
            schema:
              type: object
              properties:
                answer:
                  type: string
                citations:
                  type: array
                  items:
                    type: object
                    properties:
                      source:
                        type: string
                      page:
                        type: integer
                      chunk_id:
                        type: string
                warnings:
                  type: array
                  items:
                    type: string
                suggested_questions:
                  type: array
                  items:
                    type: string
      400:
        description: Empty question, or no documents indexed yet.
      500:
        description: Chat failed.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    provider = data.get("provider") or STATE["llm_provider"]

    if not question:
        return _error_response("Question cannot be empty.")
    if not _store.is_ready:
        return _error_response("No documents are indexed yet. Upload a document first.")

    try:
        llm = _llm_factory.create_llm(provider)

        retrieved = _retriever.retrieve(question)

        # Relevance check (Phase 2): reranking always returns its
        # "least bad" candidates even when nothing indexed actually
        # relates to the query, so grade the top_k against the
        # question before letting anything reach the answer prompt or
        # the citations shown to the user (see
        # src/retrieval/relevance.py for why this can't just be a
        # cross-encoder score threshold).
        if retrieved:
            any_relevant, relevant = check_relevance(question, retrieved, llm)
            retrieved = relevant if any_relevant else []

        context = Retriever.format_context(retrieved)
        citations = Retriever.citations(retrieved)

        system_prompt = CONFIG.get("rag", {}).get("system_prompt", "")
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        response = llm.invoke(prompt)
        answer = getattr(response, "content", str(response))

        warnings = [] if retrieved else ["No relevant passages were found for this question."]
        suggested_questions = generate_followup_questions(llm, question, answer)

        with STATE_LOCK:
            STATE["chat_history"].append({"role": "user", "content": question})
            STATE["chat_history"].append(
                {"role": "assistant", "content": answer, "citations": citations}
            )

        return jsonify(
            answer=answer, citations=citations, warnings=warnings, suggested_questions=suggested_questions
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return _error_response(f"Chat failed: {e}", 500)


# ---------------------------------------------------------------------------
# API — agent chat (plan -> tools -> synthesize -> evaluate)
# ---------------------------------------------------------------------------
@app.route("/api/agent-chat", methods=["POST"])
def api_agent_chat():
    """Ask a question (agent)
    Runs the LangGraph agent: plan -> tools (retrieval, web search, math) -> synthesize -> evaluate.
    ---
    tags:
      - Chat
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - question
            properties:
              question:
                type: string
                example: Compare total credits of both statements
              provider:
                type: string
                enum: [ollama, groq, huggingface]
                description: LLM provider. Defaults to the server's current provider.
    responses:
      200:
        description: Agent answer with citations.
        content:
          application/json:
            schema:
              type: object
              properties:
                answer:
                  type: string
                citations:
                  type: array
                  items:
                    type: object
                    properties:
                      source:
                        type: string
                      page:
                        type: integer
                      chunk_id:
                        type: string
                warnings:
                  type: array
                  items:
                    type: string
                verification_status:
                  type: string
                  nullable: true
                  description: verified / failed / unavailable / null
                suggested_questions:
                  type: array
                  items:
                    type: string
      400:
        description: Empty question.
      500:
        description: Agent chat failed.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    provider = data.get("provider") or STATE["llm_provider"]

    if not question:
        return _error_response("Question cannot be empty.")

    try:
        llm = _llm_factory.create_llm(provider)
        result = run_agent(
            question,
            provider,
            CONFIG,
            _retriever,
            llm,
            indexed_files=STATE["indexed_files"],
            thread_id=STATE["thread_id"],
            user_id=STATE["user_id"],
            memory_store=_memory_store,
        )

        with STATE_LOCK:
            STATE["chat_history"].append({"role": "user", "content": question})
            STATE["chat_history"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                }
            )

        return jsonify(**result)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return _error_response(f"Agent chat failed: {e}", 500)


# ---------------------------------------------------------------------------
# API — clear index (EXISTING button in the UI; this is its handler)
# ---------------------------------------------------------------------------
@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear the index
    Deletes the FAISS index, extraction/chunk artifacts, uploaded files and chat history.
    ---
    tags:
      - Index
    responses:
      200:
        description: Everything cleared.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
      500:
        description: Failed to clear the index.
    """
    try:
        # 1 & 2. Clear the vector index/store and any related chunk state.
        _store.clear()

        # 3. Clear the extraction-text preview artifact.
        clear_extraction_text()
        clear_chunking_text()
        clear_faiss_log()
        _clear_active_pdf_metadata()
        for entry in UPLOAD_DIR.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)

        # 4 & 5. Clear corresponding UI/server state.
        with STATE_LOCK:
            STATE.update(
                index_ready=False,
                indexed_files=[],
                chat_history=[],
                active_pdf_path=None,
                active_pdf_name=None,
                pdf_paths={},
                thread_id=uuid.uuid4().hex,
            )

        return jsonify(success=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return _error_response(f"Failed to clear index: {e}", 500)


@app.route("/api/clear-chat", methods=["POST"])
def api_clear_chat():
    """Start a new chat
    Resets chat history and the agent thread. Indexed documents and long-term memories are kept.
    ---
    tags:
      - Chat
    responses:
      200:
        description: Chat reset.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
    """
    with STATE_LOCK:
        STATE["chat_history"] = []
        STATE["thread_id"] = uuid.uuid4().hex
    return jsonify(success=True)


# ---------------------------------------------------------------------------
# API — long-term memory (view / forget what the agent has remembered
# about the user across sessions; see src/services/memory_store.py).
# Independent of chat_history/thread_id -- clearing chat does NOT clear
# these, and these do NOT clear chat.
# ---------------------------------------------------------------------------
@app.route("/api/memories", methods=["GET"])
def api_memories():
    """List long-term memories
    ---
    tags:
      - Memory
    responses:
      200:
        description: Stored memories, newest first.
        content:
          application/json:
            schema:
              type: object
              properties:
                memories:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      text:
                        type: string
                      created_at:
                        type: string
                      use_count:
                        type: integer
    """
    return jsonify(memories=_memory_store.all(STATE["user_id"]))


@app.route("/api/memories", methods=["DELETE"])
def api_memories_clear():
    """Forget all long-term memories
    ---
    tags:
      - Memory
    responses:
      200:
        description: All memories deleted.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
    """
    _memory_store.delete(STATE["user_id"])
    return jsonify(success=True)


@app.route("/api/memories/<int:memory_id>", methods=["DELETE"])
def api_memory_delete(memory_id: int):
    """Forget one long-term memory
    ---
    tags:
      - Memory
    parameters:
      - name: memory_id
        in: path
        required: true
        schema:
          type: integer
    responses:
      200:
        description: Memory deleted.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
    """
    _memory_store.delete(STATE["user_id"], memory_id=memory_id)
    return jsonify(success=True)


# ---------------------------------------------------------------------------
# Serve the currently active PDF to the in-browser viewer
# ---------------------------------------------------------------------------
@app.route("/pdf/current")
def pdf_current():
    """Get the active PDF
    ---
    tags:
      - PDF
    responses:
      200:
        description: The active PDF file.
        content:
          application/pdf:
            schema:
              type: string
              format: binary
      404:
        description: No document is currently indexed.
    """
    path = STATE["active_pdf_path"]
    if not path or not os.path.exists(path):
        return "No document is currently indexed.", 404
    directory, filename = os.path.split(path)
    return send_from_directory(directory, filename, mimetype="application/pdf")


@app.route("/pdf/<path:name>")
def pdf_by_name(name):
    """Get an indexed PDF by name
    Serves a specific indexed PDF so the viewer can switch between uploaded documents.
    ---
    tags:
      - PDF
    parameters:
      - name: name
        in: path
        required: true
        description: File name as returned by /api/upload (e.g. Barclays.PDF).
        schema:
          type: string
    responses:
      200:
        description: The requested PDF file.
        content:
          application/pdf:
            schema:
              type: string
              format: binary
      404:
        description: Not an indexed PDF.
    """
    path = STATE["pdf_paths"].get(name)
    if not path or not os.path.exists(path):
        return f"'{name}' is not an indexed PDF.", 404
    directory, filename = os.path.split(path)
    return send_from_directory(directory, filename, mimetype="application/pdf")


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "5000"))
    app.run(host=host, port=port, debug=debug_mode)