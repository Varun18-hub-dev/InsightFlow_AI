from abc import ABC, abstractmethod
from typing import Any

import fitz
from docx import Document


class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> list[dict[str, Any]]:
        pass

class PDFExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> list[dict[str, Any]]:
        results = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                results.append({
                    "content": text,
                    "page_number": page_num + 1,
                    "metadata": {"source_type": "pdf"}
                })
        return results

class DOCXExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> list[dict[str, Any]]:
        results = []
        doc = Document(file_path)
        content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if content:
            results.append({
                "content": content,
                "page_number": 1,
                "metadata": {"source_type": "docx"}
            })
        return results

class TextExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> list[dict[str, Any]]:
        results = []
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if content.strip():
                results.append({
                    "content": content,
                    "page_number": 1,
                    "metadata": {"source_type": "text"}
                })
        return results

class ExtractorFactory:
    @staticmethod
    def get_extractor(document_type: str) -> DocumentExtractor:
        if document_type == "application/pdf" or document_type.endswith(".pdf"):
            return PDFExtractor()
        elif "wordprocessingml" in document_type or document_type.endswith(".docx"):
            return DOCXExtractor()
        elif document_type.startswith("text/") or document_type.endswith((".txt", ".md")):
            return TextExtractor()
        else:
            raise ValueError(f"No extractor found for type {document_type}")
