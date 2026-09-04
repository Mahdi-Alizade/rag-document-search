from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, EMBEDDING_MODEL_NAME


class VectorStore:
    def __init__(self, collection_name: str = "document_chunks"):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, chunked_docs: List[Dict[str, Any]]) -> int:
        if not chunked_docs:
            return 0

        ids = [doc["id"] for doc in chunked_docs]
        documents = [doc["text"] for doc in chunked_docs]
        metadatas = [doc["metadata"] for doc in chunked_docs]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results and results["distances"] else [None] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return retrieved

    def count(self) -> int:
        return self.collection.count()