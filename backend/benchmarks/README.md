# InsightFlow AI benchmarks

Offline, repeatable measurements of the retrieval pipeline. No API keys, database or Pinecone
needed for the retrieval benchmark.

## Why this exists

The first evaluation in this repo (now in `docs/legacy_results/`) scored 1.0 on almost every
metric. That run used 10–15 hand-written questions over a 5-document corpus, and the
"faithfulness" score was word overlap multiplied by 1.5 and capped at 1.0. A score that cannot
go down tells you nothing, so it was replaced with this benchmark:

- a public dataset that I did not write (SQuAD v1.1 dev: 2,067 paragraphs, 48 Wikipedia articles)
- one question per paragraph, sampled with a fixed seed, so all 2,067 paragraphs are tested once
- 95% bootstrap confidence intervals on every metric
- a paired bootstrap test against a baseline on the same queries
- an error analysis that sorts every failure into a bucket

## What is measured, and how close it is to production

| Stage | Production app | In this benchmark |
|---|---|---|
| Chunking | `DocumentChunker` (800 chars, 150 overlap) | same code |
| Keyword search | Postgres full-text search (`ts_rank`) | BM25 (`rank_bm25`) as a stand-in |
| Dense search | Gemini/OpenAI embeddings + Pinecone | local sentence-transformers model, exact cosine search |
| Fusion | `rrf_fusion`, k=60 | same code |
| Reranking | `LightweightReranker` / `CrossEncoderReranker` | same code |

Ranking is scored at document (paragraph) level: chunks are collapsed to their paragraph, keeping
the best position.

## Results: SQuAD v1.1 dev, all 2,067 paragraphs

Command: `python -m benchmarks.run_retrieval_benchmark --configs bm25,bm25+light`
Full table: [`results/squad/summary.md`](results/squad/summary.md); every query's outcome:
[`results/squad/per_query.jsonl`](results/squad/per_query.jsonl).

| Config | Recall@1 | Recall@5 | MRR@10 | Answer in top 5 | p50 latency |
|---|---|---|---|---|---|
| BM25 | 0.780 [0.761, 0.797] | 0.926 [0.915, 0.937] | 0.843 [0.830, 0.856] | 0.941 | 3.3 ms |
| BM25 + lightweight reranker | 0.794 [0.776, 0.811] | 0.930 [0.919, 0.940] | 0.852 [0.839, 0.865] | 0.941 | 4.0 ms |

**What this shows**

1. **The lightweight reranker gives a small but real gain in ranking, not in coverage.**
   MRR@10 improves by +0.009 (95% CI +0.005 to +0.014, p < 0.001). Recall@5 improves by +0.003,
   but its CI includes zero (p = 0.11), so I do not claim a recall gain.
2. **The 45 true misses (2.2%) fall into two clear groups.** I read all 45 (they are in
   `per_query.jsonl`, bucket `miss`):
   - **Context-dependent questions (11 of 45)**, such as "Why did he fire the secretary?" or
     "What did this agreement do?". They were written while looking at the paragraph, so "he"
     and "this" point to nothing. Chat users ask follow-ups like this all the time, which argues
     for rewriting a follow-up question with the chat history before retrieval.
   - **Misspelled key terms**, such as "Raouliii", "platycenida", "Cypiddids", "argricultural",
     "Survivial". BM25 needs exact tokens, so one typo in the only rare word sinks the query.
     This is the clearest case where dense retrieval should help, and it is what the hybrid
     run below should confirm or refute.
3. **Some "misses" are labelling gaps.** SQuAD marks only one paragraph as correct. In 25 of the
   reranked misses, a different top-5 paragraph still contains the gold answer. Strict recall
   counts these as failures, which is why answer-in-top-5 is reported next to it.

**Caveat:** SQuAD questions share many words with their paragraph, which favours keyword search.
These numbers are an upper bound for BM25 and should not be read as "BM25 is enough". The
dense and hybrid configurations are the real test, and they need the run below.

## Not yet measured (needs a machine that can download models)

This workspace could not reach Hugging Face, so these configs have code and tests but no
reported numbers yet:

```bash
# from backend/, with the BGE-small model downloaded locally (or let it download)
python -m benchmarks.run_retrieval_benchmark \
    --configs bm25,dense,hybrid,hybrid+light,hybrid+ce \
    --embedding-model C:/models/bge-small-en-v1.5
```

The question this answers: does adding dense retrieval and RRF actually beat BM25 on this data,
and is the cross-encoder worth its latency? If hybrid does not beat BM25 significantly, that is
a result worth reporting, not hiding.

A second dataset with less word overlap (BEIR SciFact or FiQA) can be run the same way:

```bash
python -m benchmarks.run_retrieval_benchmark --dataset beir --beir-dir data/scifact --configs bm25,dense,hybrid
```

## Answer quality (end to end)

`run_generation_benchmark.py` retrieves with a chosen config, asks the configured LLM to answer,
and scores with the official SQuAD exact match and token F1. It reports F1 separately for
questions whose answer was actually retrieved, so retrieval misses and generation mistakes are
not mixed together, and it tracks how often the model says it could not find the answer.

```bash
GEMINI_API_KEY=... python -m benchmarks.run_generation_benchmark --config bm25+light --n-queries 200
```

## Tests

`tests/unit/test_benchmark_metrics.py` checks the metrics against hand-computed values, checks
that the bootstrap separates a real difference from no difference, and checks the data loaders.
