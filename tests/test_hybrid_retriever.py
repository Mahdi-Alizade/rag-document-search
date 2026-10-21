import unittest
from unittest.mock import MagicMock
from hybrid_retriever import HybridRetriever


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock()
        self.mock_bm25_retriever = MagicMock()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.mock_vector_store,
            bm25_retriever=self.mock_bm25_retriever,
            rrf_k=60
        )

    def test_search_returns_empty_when_query_is_blank(self):
        query = "   "
        results = self.hybrid_retriever.search(query)
        self.assertEqual(len(results), 0)

    def test_rrf_scoring_prioritizes_overlapping_matches(self):
        doc_shared = {
            "id": "doc_both_systems",
            "text": "Shared chunk in both dense and sparse",
            "metadata": {"source": "manual.txt"}
        }
        doc_vector_only = {
            "id": "doc_dense_only",
            "text": "Only found by vector store",
            "metadata": {"source": "manual.txt"}
        }
        doc_bm25_only = {
            "id": "doc_sparse_only",
            "text": "Only found by BM25 keyword",
            "metadata": {"source": "manual.txt"}
        }

        self.mock_vector_store.search_similar.return_value = [
            doc_shared,
            doc_vector_only
        ]
        self.mock_bm25_retriever.search.return_value = [
            doc_shared,
            doc_bm25_only
        ]

        results = self.hybrid_retriever.search("system architecture", top_k=3)

        self.assertEqual(len(results), 3)
        top_result = results[0]
        self.assertEqual(top_result["id"], "doc_both_systems")
        self.assertGreater(top_result["rrf_score"], results[1]["rrf_score"])
        self.assertIsNotNone(top_result["vector_rank"])
        self.assertIsNotNone(top_result["bm25_rank"])


if __name__ == "__main__":
    unittest.main()