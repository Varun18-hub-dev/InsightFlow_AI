import asyncio
import time
from collections.abc import AsyncIterator

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.llm.base import BaseEmbeddingProvider, BaseLLMProvider

logger = structlog.get_logger()


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        # Resolve model name
        self.model_name = (
            getattr(settings, "GEMINI_MODEL", None)
            or getattr(settings, "GEMINI_MODEL_NAME", None)
            or "gemini-flash-lite-latest"
        )

        self.api_key = settings.get_gemini_api_key()
        api_key_configured = bool(self.api_key)
        logger.info(
            "gemini_provider_init",
            configured_model=self.model_name,
            api_key_configured=api_key_configured,
        )

        self.client = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            timeout=60.0,
            max_retries=1,
        )

    def _convert_messages(self, messages: list[dict[str, str]]):
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
        return langchain_messages

    def _validate_messages(self, messages: list[dict[str, str]]):
        if not messages:
            raise ValueError("Messages list is empty.")
        if not self.api_key:
            raise ValueError("settings.GEMINI_API_KEY is not configured.")

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content or not str(content).strip():
                raise ValueError(
                    f"Message at index {idx} with role '{role}' has empty content."
                )

    @staticmethod
    def _extract_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content) if content is not None else ""

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self._validate_messages(messages)
        lc_msgs = self._convert_messages(messages)
        start = time.time()
        logger.info(
            "gemini_chat_invoking",
            model=self.model_name,
            message_count=len(messages),
        )
        response = await self.client.ainvoke(lc_msgs)
        text_content = self._extract_text(response.content)
        elapsed = round(time.time() - start, 3)
        logger.info(
            "gemini_chat_completed",
            model=self.model_name,
            elapsed_seconds=elapsed,
            response_len=len(text_content),
        )
        return text_content

    async def chat_stream(self, messages: list[dict[str, str]], **kwargs) -> AsyncIterator[str]:
        self._validate_messages(messages)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        user_msgs = [m for m in messages if m.get("role") != "system"]

        logger.info(
            "gemini_stream_invoking",
            model=self.model_name,
            total_messages=len(messages),
            system_msg_len=len(system_msgs[0].get("content", "")) if system_msgs else 0,
            user_msg_len=len(user_msgs[0].get("content", "")) if user_msgs else 0,
            api_key_configured=bool(settings.GEMINI_API_KEY),
        )

        lc_msgs = self._convert_messages(messages)

        # Log before calling astream
        logger.info(
            "gemini_astream_calling",
            model=self.model_name,
            lc_msg_count=len(lc_msgs),
        )
        stream = self.client.astream(lc_msgs)
        # Log after obtaining astream
        logger.info(
            "gemini_astream_iterator_obtained",
            model=self.model_name,
        )

        STREAM_TIMEOUT_SECONDS = 60.0
        stream_iter = stream.__aiter__()
        chunk_count = 0
        stream_start = time.time()

        # Log before entering loop
        logger.info("gemini_stream_loop_entering", model=self.model_name)

        while True:
            try:
                chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=STREAM_TIMEOUT_SECONDS,
                )
            except StopAsyncIteration:
                break
            except TimeoutError as e:
                logger.error(
                    "gemini_stream_timeout",
                    timeout_type="chunk_timeout",
                    model=self.model_name,
                    chunks_received=chunk_count,
                    elapsed_seconds=round(time.time() - stream_start, 3),
                )
                raise TimeoutError(
                    f"Gemini streaming timed out after {STREAM_TIMEOUT_SECONDS}s"
                ) from e
            except Exception as e:
                err_str = str(e)
                if "timeout" in err_str.lower() or "deadline" in err_str.lower():
                    logger.error(
                        "gemini_stream_timeout",
                        timeout_type="client_timeout",
                        model=self.model_name,
                        chunks_received=chunk_count,
                        elapsed_seconds=round(time.time() - stream_start, 3),
                    )
                raise

            chunk_count += 1
            text_part = self._extract_text(chunk.content) if chunk.content else ""
            if chunk_count == 1:
                first_chunk_elapsed = round(time.time() - stream_start, 3)
                logger.info(
                    "gemini_first_chunk_received",
                    model=self.model_name,
                    elapsed_seconds=first_chunk_elapsed,
                    content_len=len(text_part),
                )

            if text_part:
                yield text_part

        logger.info(
            "gemini_astream_completed",
            model=self.model_name,
            total_chunks=chunk_count,
            elapsed_seconds=round(time.time() - stream_start, 3),
        )

    def get_model_name(self) -> str:
        return self.model_name


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        api_key = settings.get_gemini_api_key()
        # gemini-embedding-001 is the current stable embedding model
        self.client = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

    async def embed_query(self, text: str) -> list[float]:
        return await self.client.aembed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.client.aembed_documents(texts)

    def get_dimension(self) -> int:
        return 3072

    async def run_diagnostic(self) -> dict:
        """Temporary diagnostic testing Gemini embedding independently of chat generation."""
        test_doc = "This is a test document about machine learning."
        test_query = "What is machine learning?"
        model_name = "models/gemini-embedding-001"
        report = {}

        # 1. Document embedding diagnostic
        start_doc = time.time()
        try:
            doc_embeddings = await self.embed_documents([test_doc])
            latency_doc = round(time.time() - start_doc, 3)
            dim_doc = len(doc_embeddings[0]) if doc_embeddings and doc_embeddings[0] else 0
            logger.info(
                "gemini_embedding_diagnostic_document",
                model=model_name,
                status="SUCCESS",
                dimension=dim_doc,
                latency_seconds=latency_doc,
            )
            report["document"] = {
                "model": model_name,
                "status": "SUCCESS",
                "dimension": dim_doc,
                "latency_seconds": latency_doc,
                "error": None,
            }
        except Exception as e:
            latency_doc = round(time.time() - start_doc, 3)
            err_type = type(e).__name__
            err_msg = str(e)
            logger.error(
                "gemini_embedding_diagnostic_document",
                model=model_name,
                status="FAILURE",
                error_type=err_type,
                error_message=err_msg,
                latency_seconds=latency_doc,
            )
            report["document"] = {
                "model": model_name,
                "status": "FAILURE",
                "dimension": 0,
                "latency_seconds": latency_doc,
                "error_type": err_type,
                "error_message": err_msg,
            }

        # 2. Query embedding diagnostic (RETRIEVAL_QUERY)
        start_query = time.time()
        try:
            query_embedding = await self.embed_query(test_query)
            latency_query = round(time.time() - start_query, 3)
            dim_query = len(query_embedding) if query_embedding else 0
            logger.info(
                "gemini_embedding_diagnostic_query",
                model=model_name,
                task_type="RETRIEVAL_QUERY",
                status="SUCCESS",
                dimension=dim_query,
                latency_seconds=latency_query,
            )
            report["query"] = {
                "model": model_name,
                "task_type": "RETRIEVAL_QUERY",
                "status": "SUCCESS",
                "dimension": dim_query,
                "latency_seconds": latency_query,
                "error": None,
            }
        except Exception as e:
            latency_query = round(time.time() - start_query, 3)
            err_type = type(e).__name__
            err_msg = str(e)
            logger.error(
                "gemini_embedding_diagnostic_query",
                model=model_name,
                task_type="RETRIEVAL_QUERY",
                status="FAILURE",
                error_type=err_type,
                error_message=err_msg,
                latency_seconds=latency_query,
            )
            report["query"] = {
                "model": model_name,
                "task_type": "RETRIEVAL_QUERY",
                "status": "FAILURE",
                "dimension": 0,
                "latency_seconds": latency_query,
                "error_type": err_type,
                "error_message": err_msg,
            }

        return report
