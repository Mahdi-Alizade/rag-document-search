# RAG Document Search Engine

A lightweight, local-first Retrieval-Augmented Generation (RAG) system built in Python. This project allows users to ingest local documents (`.txt` and `.pdf`), generate dense vector embeddings, store them in ChromaDB, and execute grounded semantic queries with citations.

---

## Architecture Overview

User Query ──► Sentence-Transformers (Embedding)
│
▼
[Raw Documents] ──► Text Chunker ──► ChromaDB Vector Store
│
Top-k Relevant Chunks
│
▼
Context + Prompt Template ───────► OpenAI LLM ──► Grounded Answer + Citations


- **Document Ingestion:** Reads `.txt` and `.pdf` files from the `data/` directory.
- **Chunking Strategy:** Sliding window chunking with configurable chunk size and overlap.
- **Vector Storage:** Persistent local ChromaDB using `all-MiniLM-L6-v2` embeddings.
- **Retrieval & Synthesis:** Similarity search with distance filtering, combined with LLM prompt context injection.

---

## Project Structure

```text
rag-document-search/
├── data/                  # Source documents directory (.txt, .pdf)
├── chroma_db/             # Local persistent vector database (git-ignored)
├── .env.example           # Template for environment variables
├── .gitignore             # Secrets, database, and virtual environment ignore rules
├── requirements.txt       # Project dependencies
├── config.py              # Central application configuration
├── document_loader.py     # File parsing and text chunking
├── vector_store.py        # ChromaDB wrapper with distance thresholding
├── rag_engine.py          # Context builder and LLM synthesis
└── main.py                # CLI application entrypoint
Getting Started
1. Prerequisites
Python 3.10+

Git

2. Environment Setup
Clone the repository and create an isolated virtual environment:

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
3. Environment Variables
Copy .env.example to .env:

PowerShell
Copy-Item .env.example .env
Edit .env and configure your credentials:

Code snippet
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
Note: If an API key is not provided, the engine operates in semantic-only mode, returning the top retrieved document chunks directly.

Usage
Place your .txt or .pdf files inside the data/ folder.

Run the application:

PowerShell
python main.py
Ask natural language questions in the CLI interface. Type exit or quit to leave the interactive prompt.
