import os
from typing import List, Dict, Any
from openai import OpenAI


class RAGEngine:
    def __init__(self, retriever, model_name: str = "gpt-4o-mini"):
        self.retriever = retriever
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = None

        if self.api_key and len(self.api_key.strip()) > 0:
            self.client = OpenAI(api_key=self.api_key)
        else:
            print("Notice: OPENAI_API_KEY not found. Operating in offline semantic mode.")

    def format_context_passages(self, chunks: List[Dict[str, Any]]) -> str:
        formatted_blocks = []

        for index, item in enumerate(chunks):
            passage_text = item.get("text", "").strip()
            meta = item.get("metadata", {})
            source_file = meta.get("source", "unknown")
            chunk_idx = meta.get("chunk_index", index)

            header = f"[Document: {source_file} | Chunk: {chunk_idx}]"
            block = f"{header}\n{passage_text}"
            formatted_blocks.append(block)

        full_context = "\n\n".join(formatted_blocks)
        return full_context

    def generate_prompt(self, query: str, context: str) -> str:
        prompt_template = (
            "You are a factual, concise knowledge assistant.\n"
            "Answer the user question strictly using the provided context below.\n"
            "If the information is not contained in the context, clearly state that you don't know.\n"
            "Always cite the source document name when mentioning specific facts.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return prompt_template

    def ask(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "Query is empty.",
                "sources": []
            }

        retrieved_candidates = self.retriever.search(clean_query, top_k=top_k)

        if not retrieved_candidates:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": []
            }

        context_string = self.format_context_passages(retrieved_candidates)

        if not self.client:
            offline_answer = "Semantic Mode (No LLM API Key):\n" + context_string
            return {
                "answer": offline_answer,
                "sources": retrieved_candidates
            }

        prompt_message = self.generate_prompt(clean_query, context_string)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on documents."},
                    {"role": "user", "content": prompt_message}
                ],
                temperature=0.2
            )
            answer_text = response.choices[0].message.content.strip()
        except Exception as err:
            answer_text = f"LLM Generation Error: {err}\n\nFallback context:\n{context_string}"

        return {
            "answer": answer_text,
            "sources": retrieved_candidates
        }