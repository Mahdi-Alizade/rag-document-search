import pytest
from vector_store import VectorStore


@pytest.fixture(scope="module")
def vector_store():
    # Use a dedicated isolated test collection
    store = VectorStore(collection_name="test_collection")
    store.reset_collection()
    yield store
    store.reset_collection()


def test_add_and_query_documents(vector_store):
    sample_chunks = [
        {
            "id": "doc_1_chunk_0",
            "text": "The Python language was created by Guido van Rossum.",
            "metadata": {"source": "python_history.txt", "chunk_index": 0}
        },
        {
            "id": "doc_2_chunk_0",
            "text": "Solar energy is harvested using photovoltaic solar panels.",
            "metadata": {"source": "green_energy.txt", "chunk_index": 0}
        }
    ]

    added = vector_store.add_documents(sample_chunks)
    assert added == 2
    assert vector_store.count() >= 2

    # Query specifically for python creator
    results = vector_store.query(query_text="Who created Python?", n_results=1)

    assert len(results) == 1
    assert "Guido van Rossum" in results[0]["text"]
    assert results[0]["metadata"]["source"] == "python_history.txt"


def test_query_distance_threshold(vector_store):
    # Query completely unrelated text with strict threshold
    results = vector_store.query(
        query_text="Underwater marine biology and coral reefs",
        n_results=2,
        max_distance=0.1
    )
    # Because distance is large, it should filter out unrelated items
    assert len(results) == 0