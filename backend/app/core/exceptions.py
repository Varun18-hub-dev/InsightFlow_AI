
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    error: ErrorDetail

class InsightFlowException(Exception):
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)

class DocumentProcessingError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR")

class RetrievalError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "RETRIEVAL_ERROR")

class EmbeddingError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "EMBEDDING_ERROR")

class AuthenticationError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "AUTHENTICATION_ERROR")

class AuthorizationError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "AUTHORIZATION_ERROR")

class ValidationError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")

class ExternalServiceError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR")

class SQLValidationError(InsightFlowException):
    def __init__(self, message: str):
        super().__init__(message, "SQL_VALIDATION_ERROR")
