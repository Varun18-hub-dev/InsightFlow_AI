from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings
from app.llm.base import BaseEmbeddingProvider, BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        self.client = ChatOpenAI(model=self.model_name, api_key=settings.OPENAI_API_KEY)

    def _convert_messages(self, messages: list[dict[str, str]]):
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        lc_msgs = self._convert_messages(messages)
        response = await self.client.ainvoke(lc_msgs, **kwargs)
        return response.content

    async def chat_stream(self, messages: list[dict[str, str]], **kwargs) -> AsyncIterator[str]:
        lc_msgs = self._convert_messages(messages)
        async for chunk in self.client.astream(lc_msgs, **kwargs):
            yield chunk.content

    def get_model_name(self) -> str:
        return self.model_name

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        self.client = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

    async def embed_query(self, text: str) -> list[float]:
        return await self.client.aembed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.client.aembed_documents(texts)

    def get_dimension(self) -> int:
        return 1536
