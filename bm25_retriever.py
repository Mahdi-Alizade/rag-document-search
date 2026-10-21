import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.corpus_chunks = []
        self.tokenized_corpus = []
        self.bm25_engine = None

    def _tokenize(self, text: str) -> List[str]:
        cleaned_text = text.lower()
        tokens = re.findall(r"\b\w+\b", cleaned_text)
        return tokens

    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        self.corpus_chunks = chunks
        self.tokenized_corpus = []

        for item in chunks:
            raw_content = item.get("text", "")
            chunk_tokens = self._tokenize(raw_content)
            self.tokenized_corpus.append(chunk_tokens)

        if len(self.tokenized_corpus) > 0:
            self.bm25_engine = BM25Okapi(self.tokenized_corpus)
            print(f"BM25 index built with {len(self.tokenized_corpus)} entries.")
        else:
            self.bm25_engine = None
            print("BM25 warning: No tokens found to index.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25_engine or not self.corpus_chunks:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        query_tokens = self._tokenize(clean_query)
        if not query_tokens:
            return []

        scores = self.bm25_engine.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True
        )

        final_matches = []
        selected_indices = ranked_indices[:top_k]

        for position, idx in enumerate(selected_indices):
            score_val = float(scores[idx])
            if score_val <= 0:
                continue

            matched_chunk = self.corpus_chunks[idx]
            result_item = {
                "id": matched_chunk["id"],
                "text": matched_chunk["text"],
                "metadata": matched_chunk.get("metadata", {}),
                "score": score_val,
                "rank": position + 1
            }
            final_matches.append(result_item)

        return final_matches