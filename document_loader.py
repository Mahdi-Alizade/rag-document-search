from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoader:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_text_file(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_pdf_file(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        pages_text = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                pages_text.append(extracted)
        return "\n".join(pages_text)

    def load_documents(self) -> List[Dict[str, Any]]:
        documents = []
        supported_extensions = {".txt", ".pdf"}

        for file_path in self.data_dir.iterdir():
            if file_path.suffix.lower() not in supported_extensions:
                continue

            try:
                if file_path.suffix.lower() == ".txt":
                    content = self.load_text_file(file_path)
                elif file_path.suffix.lower() == ".pdf":
                    content = self.load_pdf_file(file_path)
                else:
                    content = ""

                if content.strip():
                    documents.append({
                        "source": file_path.name,
                        "content": content
                    })
            except Exception as e:
                print(f"Error reading file {file_path.name}: {e}")

        return documents

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += (chunk_size - overlap)

        return chunks

    def process(self) -> List[Dict[str, Any]]:
        raw_documents = self.load_documents()
        chunked_documents = []

        for doc in raw_documents:
            chunks = self.chunk_text(doc["content"])
            for idx, chunk in enumerate(chunks):
                chunked_documents.append({
                    "id": f"{doc['source']}_chunk_{idx}",
                    "text": chunk,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_index": idx
                    }
                })

        return chunked_documents