from typing import List, Dict, Any


class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def _compute_rrf_scores(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        combined_scores = {}
        chunk_map = {}

        # 1. Process vector results
        for rank_idx, item in enumerate(vector_results):
            chunk_id = item["id"]
            current_rank = rank_idx + 1
            score_increment = 1.0 / (self.rrf_k + current_rank)

            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = 0.0
                chunk_map[chunk_id] = {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "vector_rank": current_rank,
                    "bm25_rank": None
                }
            
            combined_scores[chunk_id] = combined_scores[chunk_id] + score_increment

        # 2. Process BM25 lexical results
        for rank_idx, item in enumerate(bm25_results):
            chunk_id = item["id"]
            current_rank = rank_idx + 1
            score_increment = 1.0 / (self.rrf_k + current_rank)

            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = 0.0
                chunk_map[chunk_id] = {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "vector_rank": None,
                    "bm25_rank": current_rank
                }
            else:
                chunk_map[chunk_id]["bm25_rank"] = current_rank

            combined_scores[chunk_id] = combined_scores[chunk_id] + score_increment

        # 3. Sort by aggregated RRF score
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda cid: combined_scores[cid],
            reverse=True
        )

        ranked_output = []
        for position, cid in enumerate(sorted_chunk_ids):
            record = chunk_map[cid]
            record["rrf_score"] = combined_scores[cid]
            record["combined_rank"] = position + 1
            ranked_output.append(record)

        return ranked_output

    def search(self, query: str, top_k: int = 5, candidate_pool_size: int = 15) -> List[Dict[str, Any]]:
        clean_query = query.strip()
        if not clean_query:
            return []

        # Retrieve candidates from dense vector store
        vector_candidates = self.vector_store.search_similar(
            clean_query,
            top_k=candidate_pool_size
        )

        # Retrieve candidates from sparse BM25 engine
        bm25_candidates = self.bm25_retriever.search(
            clean_query,
            top_k=candidate_pool_size
        )

        # Fuse rankings using Reciprocal Rank Fusion
        merged_rankings = self._compute_rrf_scores(
            vector_results=vector_candidates,
            bm25_results=bm25_candidates
        )

        final_selection = merged_rankings[:top_k]
        return final_selection