import os
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.llm.base import BaseEmbeddingProvider, BaseLLMProvider

# Suppress the GOOGLE_API_KEY conflict warning — prefer our explicit key
os.environ.pop("GOOGLE_API_KEY", None)


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        # Use gemini-flash-latest
        self.model_name = getattr(settings, "GEMINI_MODEL", "gemini-flash-latest")
        if self.model_name == "gemini-2.5-flash":
            self.model_name = "gemini-flash-latest"
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=settings.GEMINI_API_KEY,
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

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        lc_msgs = self._convert_messages(messages)
        response = await self.client.ainvoke(lc_msgs)
        return response.content

    async def chat_stream(self, messages: list[dict[str, str]], **kwargs) -> AsyncIterator[str]:
        lc_msgs = self._convert_messages(messages)
        async for chunk in self.client.astream(lc_msgs):
            if chunk.content:
                yield chunk.content

    def get_model_name(self) -> str:
        return self.model_name


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        os.environ.pop("GOOGLE_API_KEY", None)
        # gemini-embedding-001 is the current stable embedding model
        self.client = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY,
        )

    async def embed_query(self, text: str) -> list[float]:
        return await self.client.aembed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.client.aembed_documents(texts)

    def get_dimension(self) -> int:
        return 3072
