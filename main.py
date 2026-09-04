import sys
from document_loader import DocumentLoader
from vector_store import VectorStore
from rag_engine import RAGEngine


def index_documents(vector_store: VectorStore) -> None:
    print("Scanning data directory for documents...")
    loader = DocumentLoader()
    chunked_docs = loader.process()

    if not chunked_docs:
        print("No documents found in the 'data' directory.")
        print("Place .txt or .pdf files in the 'data' folder and run again.")
        return

    print(f"Extracted {len(chunked_docs)} chunks from documents.")
    added_count = vector_store.add_documents(chunked_docs)
    print(f"Successfully indexed {added_count} chunks into ChromaDB.")


def interactive_cli(rag_engine: RAGEngine) -> None:
    print("\n" + "=" * 50)
    print("RAG Document Search Ready")
    print("Type your question below (or type 'exit' or 'quit' to stop):")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("User Query: ").strip()
            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                print("Exiting application.")
                sys.exit(0)

            print("\nSearching documents and generating response...\n")
            result = rag_engine.answer_question(query)

            print("--- Answer ---")
            print(result["answer"])
            print("\n--- Sources ---")
            for source in result["sources"]:
                print(f"- {source}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {e}\n")


def main() -> None:
    vector_store = VectorStore()

    if vector_store.count() == 0:
        print("Vector database is empty. Performing initial ingestion...")
        index_documents(vector_store)
    else:
        print(f"Loaded existing database with {vector_store.count()} indexed chunks.")
        reindex = input("Do you want to re-scan and index documents? (y/N): ").strip().lower()
        if reindex == "y":
            index_documents(vector_store)

    engine = RAGEngine(vector_store=vector_store)
    interactive_cli(engine)


if __name__ == "__main__":
    main()