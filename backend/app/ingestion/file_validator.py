import os
import re

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import ValidationError


class FileValidator:
    ALLOWED_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown"
    }
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    @classmethod
    def validate_file(cls, file: UploadFile) -> None:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValidationError(f"File size exceeds the limit of {settings.MAX_FILE_SIZE_MB}MB.")

        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in cls.ALLOWED_EXTENSIONS:
            raise ValidationError(f"File extension {ext} not allowed.")

        if file.content_type not in cls.ALLOWED_TYPES:
            raise ValidationError(f"File content type {file.content_type} not allowed.")

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        s = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        return s.strip('_')
