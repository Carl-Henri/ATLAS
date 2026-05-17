# ATLAS — Agentic RAG System for Engineering Documents

An AI-powered Retrieval-Augmented Generation (RAG) platform developed during an R&D internship at **Thales AVS**. ATLAS enables engineers to query large corpora of technical documents (PDFs, HTML pages, Excel spreadsheets, plain text) using natural language, with full support for figures, tables, and multi-modal content.

## Features

- **Hybrid search** — combines dense vector search (ChromaDB + BGE-M3 embeddings) with sparse keyword search (Whoosh BM25) and a cross-encoder reranker for high-precision retrieval
- **Agentic workflows** — LangGraph-based agents that decompose complex engineering queries into sub-tasks (requirement analysis, test coverage, traceability, etc.)
- **Visual RAG** — ColPali-powered page-level retrieval for documents where layout and figures matter
- **Multi-format ingestion** — automated pipeline for PDF (text + figures via Docling), HTML, Excel, and plain text
- **Gradio web interface** — interactive UI to query documents, browse sources, and inspect retrieved chunks
- **RAGAS evaluation** — automated evaluation of retrieval quality and answer faithfulness
- **Glossary management** — domain-specific terminology lookup integrated into the retrieval pipeline

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| LLMs | Mistral (small / medium / large), GPT-4.1 |
| Embeddings | BGE-M3 (local, via HuggingFace) |
| Vector store | ChromaDB |
| Keyword search | Whoosh (BM25) |
| Reranker | Cross-encoder (local) |
| Agentic framework | LangChain + LangGraph |
| Visual retrieval | ColPali |
| PDF parsing | Docling, PyMuPDF |
| UI | Gradio |
| Evaluation | RAGAS |

## Project Structure

```
ATLAS/
├── my_rag.py                      # Core RAG pipeline (retrieve + answer)
├── my_agentic_rag.py              # Agentic RAG with multi-step reasoning
├── agentic_workflow.py            # High-level workflow orchestration
├── hybrid_search.py               # Hybrid dense + sparse retrieval
├── interface_gradio.py            # Web UI
├── ragas_evaluation.py            # Retrieval evaluation
├── manage_glossary.py             # Domain glossary integration
├── visual_rag_on_sdd.py           # Visual RAG on system design documents
├── process_pdf_documents/         # PDF ingestion pipeline
├── process_html_documents/        # HTML ingestion
├── process_excel_documents/       # Excel ingestion
├── workflows/                     # Specialized engineering workflows
│   ├── requirement_analysis.py
│   ├── workflow_HLR_analysis.py
│   ├── workflow_HLT_coverage.py
│   └── ...
└── requirements.txt
```

## Context

Built as part of a generative AI R&D engineering internship at **Thales AVS** (Valence, France), April 2024 – February 2026. The system was designed to assist engineers in navigating and querying proprietary technical documentation, reducing the time spent on manual document search.
