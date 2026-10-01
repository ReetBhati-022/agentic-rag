# Agentic RAG Project Structure

## `src/agent/`
- `graph.py` → LangGraph ka main agent workflow.
- `state.py` → Agent ka state/data store karega.
- `nodes.py` → Agent ke workflow steps aur tool-calling logic handle karega.

## `src/router/`
- `document_router.py` → Document type detect karke appropriate extraction/parser route select karega.

## `src/processing/`
- `text_extractor.py` → Digital PDFs se text aur positional information extract karega.
- `ocr_extractor.py` → Scanned PDFs/images ke liye OCR extraction handle karega.
- `layout.py` → Common layout layer; extracted text ke bbox, rows, columns aur reading order ko standardize karega.
- `field_association.py` → Labels aur values ko positional/layout information ke basis par associate karega.
- `text_normalizer.py` → Verified extracted data ko common `NormalizedDocument` format mein convert karega.

## `src/parsers/`
- `bank_statement.py` → Bank statements ke liye dedicated `BankStatementParser`.
- `structured_document.py` → Invoice, receipt, application form aur other structured documents ke liye structured extraction handle karega.

## `src/chunking/`
- `chunker.py` → `NormalizedDocument` ko metadata-preserving chunks mein divide karega.

## `src/retrieval/`
- `retriever.py` → Relevant document chunks retrieve karega.
- `faiss_store.py` → Embeddings ko FAISS mein store aur search karega.

## `src/tools/`
- `retrieval_tool.py` → Uploaded documents/FAISS se relevant information retrieve karega.
- `web_search.py` → Exa ke through current web information search karega.
- `math_tool.py` → Mathematical calculations ke liye calculator tool.

## Root Files
- `app.py` → UI / application entry point.
- `config.yaml` → Project settings and configurations.
- `requirements.txt` → Required Python dependencies.
- `.env` → API keys and environment variables.

## Main Flow

### Document Ingestion

```text
User Upload
    ↓
Document Router
    ↓
Document Type Detection
    ↓
┌──────────────────────────────────────────────────┐
│ Digital PDF       → Docling / PyMuPDF           │
│ Structured Form   → PP-StructureV3              │
│ Bank Statement    → BankStatementParser         │
│ Scanned PDF/Image → PP-StructureV3 / PaddleOCR  │
└──────────────────────────────────────────────────┘
    ↓
Raw Text + Positional Information
    ↓
Common Layout Layer
    ↓
Row / Column / Reading Order
    ↓
Field Association
    ↓
Verification
    ↓
Normalization
    ↓
NormalizedDocument
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS