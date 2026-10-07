import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "InsightFlow AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    ENVIRONMENT: str = "development"
    FRONTEND_URL: str = ""
    UPLOAD_DIR: str = "uploads"

    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/dbname"
    REDIS_URL: str = "redis://localhost:6379/0"

    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "insightflow"
    PINECONE_INDEX_NAME: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1"

    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    GEMINI_MODEL_NAME: str = "gemini-flash-lite-latest"
    LLM_PROVIDER: str = "gemini"
    EMBEDDING_PROVIDER: str = "gemini"

    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_PROJECT: str = "insightflow-ai"

    MLFLOW_TRACKING_URI: str = ""

    SECRET_KEY: str = "supersecretkey"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    MAX_FILE_SIZE_MB: int = 50
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 6
    RERANKER_TYPE: str = "lightweight"
    CACHE_TTL_SECONDS: int = 3600

    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    def get_allowed_origins(self) -> list[str]:
        origins = list(self.ALLOWED_ORIGINS) if isinstance(self.ALLOWED_ORIGINS, list) else [str(self.ALLOWED_ORIGINS)]
        if self.FRONTEND_URL and self.FRONTEND_URL not in origins:
            origins.append(self.FRONTEND_URL.rstrip("/"))
        return origins

    def get_pinecone_index(self) -> str:
        return self.PINECONE_INDEX_NAME or self.PINECONE_INDEX or "insightflow"

    def get_gemini_api_key(self) -> str:
        return (
            os.environ.get("GOOGLE_API_KEY", "")
            or self.GOOGLE_API_KEY
            or self.GEMINI_API_KEY
            or os.environ.get("GEMINI_API_KEY", "")
        )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
