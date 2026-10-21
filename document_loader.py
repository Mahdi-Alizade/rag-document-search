import os
import hashlib
from typing import List, Dict, Any
from pypdf import PdfReader


class DocumentLoader:
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory

    def _generate_doc_id(self, file_path: str) -> str:
        base_name = os.path.basename(file_path)
        hasher = hashlib.md5()
        hasher.update(base_name.encode("utf-8"))
        unique_hash = hasher.hexdigest()[:8]
        return f"{base_name}_{unique_hash}"

    def read_text_file(self, full_path: str) -> str:
        text_content = ""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        except UnicodeDecodeError:
            with open(full_path, "r", encoding="latin-1") as f:
                text_content = f.read()
        except Exception as err:
            print(f"Error reading text file {full_path}: {err}")
            text_content = ""
        return text_content

    def read_pdf_file(self, full_path: str) -> str:
        extracted_text = ""
        try:
            reader = PdfReader(full_path)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text = extracted_text + "\n" + page_text
        except Exception as err:
            print(f"Failed to read PDF file {full_path}: {err}")
            extracted_text = ""
        return extracted_text

    def load_raw_documents(self) -> List[Dict[str, Any]]:
        raw_docs = []
        if not os.path.exists(self.data_directory):
            print(f"Data directory not found: {self.data_directory}")
            return raw_docs

        for file_name in os.listdir(self.data_directory):
            file_path = os.path.join(self.data_directory, file_name)
            
            if not os.path.isfile(file_path):
                continue

            content = ""
            if file_name.endswith(".txt"):
                content = self.read_text_file(file_path)
            elif file_name.endswith(".pdf"):
                content = self.read_pdf_file(file_path)
            else:
                continue

            cleaned_content = content.strip()
            if len(cleaned_content) > 0:
                doc_record = {
                    "source": file_name,
                    "doc_id": self._generate_doc_id(file_name),
                    "content": cleaned_content
                }
                raw_docs.append(doc_record)

        return raw_docs

    def chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        chunks = []
        if not content:
            return chunks

        start_index = 0
        text_length = len(content)

        while start_index < text_length:
            end_index = start_index + chunk_size
            chunk_slice = content[start_index:end_index]
            
            clean_slice = chunk_slice.strip()
            if clean_slice:
                chunks.append(clean_slice)

            if end_index >= text_length:
                break

            start_index = start_index + (chunk_size - overlap)

        return chunks

    def process_all_documents(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        all_processed_chunks = []
        loaded_docs = self.load_raw_documents()

        for doc in loaded_docs:
            raw_text = doc["content"]
            source_file = doc["source"]
            base_doc_id = doc["doc_id"]

            pieces = self.chunk_document(raw_text, chunk_size=chunk_size, overlap=overlap)
            total_chunks = len(pieces)

            for idx, chunk_text in enumerate(pieces):
                metadata = {
                    "source": source_file,
                    "doc_id": base_doc_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "char_count": len(chunk_text)
                }

                chunk_entry = {
                    "id": f"{base_doc_id}_chunk_{idx}",
                    "text": chunk_text,
                    "metadata": metadata
                }
                all_processed_chunks.append(chunk_entry)

        return all_processed_chunks