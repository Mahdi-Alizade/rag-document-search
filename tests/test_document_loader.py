import pytest
from document_loader import DocumentLoader


@pytest.fixture
def loader(tmp_path):
    return DocumentLoader(data_dir=tmp_path)


def test_chunk_text_empty_input(loader):
    chunks = loader.chunk_text("", chunk_size=100, overlap=20)
    assert chunks == []


def test_chunk_text_smaller_than_chunk_size(loader):
    text = "This is a short sample sentence."
    chunks = loader.chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_with_overlap(loader):
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = loader.chunk_text(text, chunk_size=10, overlap=2)
    
    assert len(chunks) > 1
    # Check overlap between consecutive chunks
    assert chunks[0][-2:] == chunks[1][:2]


def test_process_reads_text_files(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Sentence one. Sentence two. Sentence three.", encoding="utf-8")

    loader = DocumentLoader(data_dir=tmp_path)
    processed = loader.process()

    assert len(processed) > 0
    assert processed[0]["metadata"]["source"] == "sample.txt"
    assert "Sentence" in processed[0]["text"]