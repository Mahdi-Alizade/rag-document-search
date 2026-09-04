from typing import List, Dict, Any
from openai import OpenAI, AuthenticationError

from config import OPENAI_API_KEY, OPENAI_MODEL
from vector_store import VectorStore


class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        is_placeholder = not OPENAI_API_KEY or "your_openai_api_key" in OPENAI_API_KEY.lower()
        self.client = OpenAI(api_key=OPENAI_API_KEY) if not is_placeholder else None
        self.model = OPENAI_MODEL

    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc.get("metadata", {}).get("source", "Unknown Source")
            context_parts.append(f"[Document {idx} - Source: {source}]\n{doc['text']}")
        return "\n\n".join(context_parts)

    def answer_question(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        retrieved_docs = self.vector_store.query(query_text=question, n_results=n_results)

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found in the database.",
                "sources": []
            }

        context = self.build_context(retrieved_docs)
        sources = [doc.get("metadata", {}).get("source", "Unknown") for doc in retrieved_docs]

        if not self.client:
            return {
                "answer": "[API Key not configured or placeholder detected]\n\nRetrieved Relevant Segments:\n\n" + context,
                "sources": list(set(sources))
            }

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context pieces to answer the question. "
            "If you do not know the answer based on the context, say that you do not know. "
            "Keep the answer concise, accurate, and faithful to the context."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            answer_text = response.choices[0].message.content.strip()
        except AuthenticationError:
            answer_text = (
                "[Invalid OpenAI API Key]\n\n"
                "Falling back to direct semantic retrieval results:\n\n" + context
            )

        return {
            "answer": answer_text,
            "sources": list(set(sources))
        }