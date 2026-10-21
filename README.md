# Hybrid RAG Document Search Engine

A local-first, modular Retrieval-Augmented Generation (RAG) platform written in Python. The system merges dense semantic vector retrieval (via ChromaDB and Sentence-Transformers) with sparse lexical keyword matching (BM25Okapi) using Reciprocal Rank Fusion (RRF), providing grounded answers with deterministic source citations.

---

## Architectural Flow

User Query
    │
    ├───► Dense Semantic Retriever (ChromaDB + all-MiniLM-L6-v2) ──► Candidate Set A
    │
    └───► Sparse Lexical Retriever (BM25 Okapi Engine)            ──► Candidate Set B
                                                                         │
                                                                         ▼
Context Assembly ◄─── Reciprocal Rank Fusion (RRF Scoring) ◄─────────────┘
      │
      ▼
Prompt Template ───► OpenAI LLM (or Offline Direct Context Mode) ──► Grounded Answer + Citations

---

## Key Features

- **Hybrid Retrieval System:** Combines semantic similarity with sparse keyword searches to effectively catch specialized domain codes, technical terms, and high-level concepts.
- **Reciprocal Rank Fusion (RRF):** Evaluates overlapping candidate positions and balances semantic and lexical signals using standard constant dampening ($k=60$).
- **Structured Metadata Tracking:** Every processed text chunk includes source filenames, relative chunk offsets, parent document hashes, and character counts.
- **Fail-Safe Offline Mode:** Operates natively in an offline direct-context output mode if no OpenAI API key is supplied in the local environment.
- **Secure File Boundaries:** All database artifacts, environment variables, virtual environments, and caches are isolated and ignored from version control.

---

## Project Structure

```text
rag-document-search/
├── data/                       # Directory for source documents (.txt, .pdf)
├── tests/                      # Automated test suites (Pytest / Unittest)
│   ├── test_document_loader.py
│   ├── test_vector_store.py
│   └── test_hybrid_retriever.py
├── chroma_db/                  # Local persistent Chroma vector database (git-ignored)
├── .env.example                # Template for environment configuration
├── .gitignore                  # Security, database, and virtual environment ignore rules
├── requirements.txt            # Python dependencies
├── config.py                   # Global configuration parameters
├── document_loader.py          # Document parsers, sanitizers, and chunking logic
├── bm25_retriever.py           # Sparse lexical search module using BM25
├── vector_store.py             # ChromaDB persistent store with batched upserts
├── hybrid_retriever.py         # Reciprocal Rank Fusion hybrid search engine
├── rag_engine.py               # Prompt generation, context builder, and LLM synthesis
└── main.py                     # Interactive CLI entrypoint
Getting Started
1. Virtual Environment Setup
Ensure PowerShell execution policies permit virtual environment activation:

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Environment Variables
Create your local environment file:

PowerShell
Copy-Item .env.example .env
Configure your credentials inside .env:

Plaintext
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-4o-mini
Note: If no API key is specified, the engine safely defaults to semantic and lexical retrieval mode without raising errors.

Running the Application
Place your knowledge documents (.txt or .pdf) into the data/ folder, then start the interface:

PowerShell
python main.py
Running Test Suites
To run all unit tests including the reciprocal rank fusion suite:

PowerShell
python -m pytest tests/