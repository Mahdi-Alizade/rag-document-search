import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, collection_name: str = "documents_collection", persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunk_items: List[Dict[str, Any]]) -> None:
        if not chunk_items:
            print("No chunks provided to insert.")
            return

        doc_ids = []
        doc_texts = []
        doc_metadatas = []

        for item in chunk_items:
            doc_ids.append(item["id"])
            doc_texts.append(item["text"])
            doc_metadatas.append(item["metadata"])

        print(f"Generating dense embeddings for {len(doc_texts)} chunks...")
        raw_embeddings = self.embedding_model.encode(doc_texts, show_progress_bar=True)
        
        embeddings_list = []
        for vec in raw_embeddings:
            embeddings_list.append(vec.tolist())

        batch_size = 64
        total_items = len(doc_ids)
        current_offset = 0

        while current_offset < total_items:
            next_offset = current_offset + batch_size
            
            batch_ids = doc_ids[current_offset:next_offset]
            batch_texts = doc_texts[current_offset:next_offset]
            batch_metadatas = doc_metadatas[current_offset:next_offset]
            batch_embeddings = embeddings_list[current_offset:next_offset]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
            
            current_offset = next_offset

        print(f"Successfully indexed {total_items} chunks in ChromaDB.")

    def search_similar(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        clean_query = query_text.strip()
        if not clean_query:
            return []

        query_vector = self.embedding_model.encode([clean_query])[0].tolist()

        search_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        retrieved_ids = search_results.get("ids", [[]])[0]
        retrieved_docs = search_results.get("documents", [[]])[0]
        retrieved_metas = search_results.get("metadatas", [[]])[0]
        retrieved_distances = search_results.get("distances", [[]])[0]

        for index in range(len(retrieved_ids)):
            item_data = {
                "id": retrieved_ids[index],
                "text": retrieved_docs[index],
                "metadata": retrieved_metas[index],
                "distance": float(retrieved_distances[index])
            }
            formatted_results.append(item_data)

        return formatted_results

    def count_records(self) -> int:
        return self.collection.count()