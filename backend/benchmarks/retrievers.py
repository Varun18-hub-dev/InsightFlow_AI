"""Offline versions of InsightFlow's retrieval stages, built on the app's own code.

The benchmark reuses the production pieces wherever they run without external services:
  - app.ingestion.chunker.DocumentChunker   (same chunk size / overlap as the app)
  - app.retrieval.rrf_fusion.rrf_fusion       (same fusion, k=60)
  - app.reranking.*                           (same lightweight and cross-encoder rerankers)

Two stages are stand-ins, and the README says so:
  - Keyword search: production uses Postgres full-text search (ts_rank). Here it is BM25
    (rank_bm25), so no database is needed.
  - Dense search: production calls Gemini/OpenAI embeddings and Pinecone. Here it is a local
    sentence-transformers model with exact cosine search, so the run is free and repeatable.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

import numpy as np

from app.ingestion.chunker import DocumentChunker
from app.retrieval.base_retriever import RetrievedChunk
from app.retrieval.rrf_fusion import rrf_fusion

_STOP = set(
    "a an and are as at be but by for from has have he in is it its of on or that the this to was "
    "were what when where which who whom why how with did does do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP]


def build_chunks(corpus: dict, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[RetrievedChunk]:
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc_id, doc in corpus.items():
        for c in chunker.chunk(doc["text"], {"doc_id": doc_id, "source": doc["title"]}):
            chunks.append(RetrievedChunk(
                id=f"{doc_id}#{c['chunk_index']}",
                content=c["content"],
                score=0.0,
                metadata=c["metadata"],
            ))
    return chunks


def collapse_to_docs(chunks: list[RetrievedChunk]) -> list[str]:
    """Chunk ranking -> document ranking, keeping each document's best position."""
    seen, out = set(), []
    for c in chunks:
        d = c.metadata["doc_id"]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


class BM25Index:
    name = "bm25"

    def __init__(self, chunks: list[RetrievedChunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.content) for c in chunks])

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:top_k]
        return [_hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(idx)]


class DenseIndex:
    name = "dense"

    def __init__(self, chunks: list[RetrievedChunk], model_name: str, cache_dir: Path | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name, device="cpu")
        # BGE models expect this instruction on queries (not on passages).
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
        )
        texts = [c.content for c in chunks]
        key = hashlib.sha1((model_name + "\n".join(c.id for c in chunks)).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"emb_{key}.npy" if cache_dir else None
        if cache and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:top_k]
        return [_hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(idx)]


class HybridIndex:
    """Mirrors HybridRetriever: run both, fuse with the app's rrf_fusion."""

    name = "hybrid_rrf"

    def __init__(self, keyword: BM25Index, dense: DenseIndex):
        self.keyword, self.dense = keyword, dense

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        return rrf_fusion([self.dense.search(query, top_k), self.keyword.search(query, top_k)])[:top_k]


class Reranked:
    """Wraps any index with one of the app's rerankers (async API, run synchronously here)."""

    def __init__(self, base, reranker, label: str):
        self.base, self.reranker, self.name = base, reranker, label

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        cands = self.base.search(query, top_k)
        return asyncio.run(self.reranker.rerank(query, cands, top_k))


def _hit(c: RetrievedChunk, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(id=c.id, content=c.content, score=score, metadata=c.metadata, rank=rank)
