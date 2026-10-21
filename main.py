import os
import sys
from dotenv import load_dotenv

from document_loader import DocumentLoader
from vector_store import VectorStore
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from rag_engine import RAGEngine


def bootstrap_system():
    load_dotenv()

    print("Initializing document ingestion pipeline...")
    loader = DocumentLoader(data_directory="data")
    chunks = loader.process_all_documents(chunk_size=500, overlap=50)

    if len(chunks) == 0:
        print("Warning: No documents were found in the data/ folder.")
        print("Please place .txt or .pdf files into the data directory.")
    else:
        print(f"Loaded and chunked {len(chunks)} total text segments.")

    print("Setting up persistent vector store...")
    vector_store = VectorStore(
        collection_name="rag_hybrid_collection",
        persist_directory="chroma_db"
    )

    if len(chunks) > 0:
        vector_store.add_documents(chunks)

    print("Setting up BM25 keyword index...")
    bm25 = BM25Retriever()
    if len(chunks) > 0:
        bm25.index_documents(chunks)

    print("Assembling hybrid rank fusion retriever...")
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        rrf_k=60
    )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    rag = RAGEngine(retriever=retriever, model_name=model_name)

    return rag


def start_cli_loop(engine: RAGEngine):
    print("\n" + "=" * 50)
    print("Hybrid RAG Knowledge Base CLI Ready.")
    print("Type your question below, or type 'exit' or 'quit' to close.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered == "exit" or lowered == "quit":
            print("Goodbye.")
            break

        print("\nSearching and formulating response...")
        result = engine.ask(user_input, top_k=4)

        answer_text = result.get("answer", "")
        sources_list = result.get("sources", [])

        print("\n--- Answer ---")
        print(answer_text)

        if len(sources_list) > 0:
            print("\n--- Retrieved Sources & Citations ---")
            for idx, item in enumerate(sources_list):
                meta = item.get("metadata", {})
                source_doc = meta.get("source", "unknown")
                chunk_idx = meta.get("chunk_index", "N/A")
                score_val = item.get("rrf_score", 0.0)
                print(f"[{idx + 1}] File: {source_doc} (Chunk #{chunk_idx}) | RRF Score: {score_val:.4f}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    app_engine = bootstrap_system()
    start_cli_loop(app_engine)