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

Command (CPU laptop, local BGE-small embeddings):

```bash
python -m benchmarks.run_retrieval_benchmark --configs bm25,dense,hybrid,hybrid+light \
    --embedding-model BAAI/bge-small-en-v1.5
```

Full table: [`results/squad/summary.md`](results/squad/summary.md); every query's outcome:
[`results/squad/per_query.jsonl`](results/squad/per_query.jsonl).

| Config | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Answer in top 5 |
|---|---|---|---|---|---|
| BM25 (keyword only) | 0.780 [0.761, 0.797] | 0.926 [0.915, 0.937] | 0.948 [0.939, 0.957] | 0.843 [0.830, 0.856] | 0.941 |
| Dense (embeddings only) | 0.684 [0.664, 0.703] | 0.895 [0.881, 0.908] | 0.940 [0.929, 0.950] | 0.773 [0.757, 0.788] | 0.917 |
| Hybrid (RRF) | 0.766 [0.748, 0.784] | 0.945 [0.935, 0.955] | **0.978** [0.971, 0.984] | 0.846 [0.833, 0.859] | **0.961** |
| Hybrid + lightweight reranker | **0.782** [0.764, 0.800] | **0.947** [0.937, 0.956] | 0.972 [0.965, 0.979] | **0.854** [0.842, 0.867] | 0.957 |

Paired difference against BM25 on the same 2,067 queries (95% bootstrap CI):

| Config | Δ Recall@5 | p | Δ MRR@10 | p |
|---|---|---|---|---|
| Dense | −0.031 [−0.047, −0.015] | < 0.001 | −0.070 [−0.087, −0.054] | < 0.001 |
| Hybrid | +0.019 [+0.009, +0.029] | < 0.001 | +0.003 [−0.008, +0.014] | 0.58 |
| Hybrid + reranker | +0.020 [+0.011, +0.030] | < 0.001 | +0.011 [+0.001, +0.022] | 0.04 |

**What this shows**

1. **Hybrid search is justified by the data.** Neither retriever alone is best: hybrid beats
   BM25 on Recall@5 by +0.019 (significant) and cuts true misses from 50 to 29. Per query, hybrid
   + reranker moved 71 questions into the top 5 that BM25 missed, and lost 29 that BM25 had.
2. **Dense alone is worse than BM25 on this dataset** (−0.031 Recall@5). SQuAD questions reuse
   the paragraph's words, which favours keyword search. This is a property of SQuAD, so it does
   not mean embeddings are useless; on a dataset with less word overlap the gap should shrink or
   flip. That is why BEIR support is included.
3. **Fusion improves coverage, the reranker restores ordering.** Plain RRF raises Recall@5 but
   not MRR (p = 0.58), and Recall@1 dips slightly. Adding the lightweight reranker brings MRR to a
   significant +0.011 over BM25 and gives the best Recall@1.
4. **Embeddings fixed the misspelled-word misses, as predicted.** "Raouliii", "platycenida",
   "Cypiddids" and "Melanchhon" were BM25 misses; all four reach the top 5 with hybrid. One
   ("argricultural") still fails.
5. **What still fails.** The 31 remaining misses are mostly context-dependent questions ("What
   did this agreement do?", "Who presided over the assembly?") that cannot be resolved without
   the paragraph. Hybrid did recover some ("Why did he fire the secretary?"), because the rest of
   the question was specific enough. Rewriting follow-up questions with chat history before
   retrieval is the next fix.
6. **Some "misses" are labelling gaps.** SQuAD marks only one paragraph as correct. For 23
   hybrid + reranker failures, another top-5 paragraph still contains the gold answer.

**Latency caveat:** dense and hybrid p50 is about 150–200 ms here because the query is embedded
on a laptop CPU. In the app, query embedding is an API call and search is in Pinecone, so these
latencies do not transfer. BM25 p50 is about 10 ms.

## Not yet measured

- **Cross-encoder reranker** (`hybrid+ce`): needs the `ms-marco-MiniLM` model downloaded.
- **A second dataset** with less word overlap, such as BEIR SciFact:

```bash
python -m benchmarks.run_retrieval_benchmark --dataset beir --beir-dir data/scifact --configs bm25,dense,hybrid
```

## Saying "I don't know" (unanswerable questions)

`run_abstention_benchmark.py` uses SQuAD 2.0 dev: 1,204 paragraphs, each with one answerable
and one **unanswerable** question. The unanswerable ones were written to look answerable from
that paragraph (same topic, same words), but the paragraph does not contain the answer. The
pipeline is the app's own, including the production prompts in `prompts/rag/` and the
`[Source: file, Page n]` context format from `app/api/chat.py`.

### Part 1: can retrieval scores detect unanswerable questions? (measured)

`python -m benchmarks.run_abstention_benchmark --configs bm25+light --retrieval-only --n-per-class 0`

| Measure (all 2,408 questions) | Value |
|---|---|
| AUROC of the top retrieval score, answerable vs unanswerable | **0.576** (0.5 = chance) |
| Unanswerable questions where the paragraph they were written on is in the 6 chunks sent to the LLM | 92.6% |
| Answerable questions whose gold answer is in those 6 chunks | 94.9% |

**What this shows:** retrieval cannot catch unanswerable questions. The top score is barely
better than a coin flip at telling them apart, and for 93% of them the LLM receives a highly
relevant, on-topic paragraph that still lacks the answer. So a "refuse if the retrieval score is
low" rule would not work, and refusing correctly depends entirely on the LLM following the
prompt. That is what Part 2 measures.

### Part 2: does the LLM refuse? (needs a key; not yet run)

```bash
python -m benchmarks.run_abstention_benchmark --configs hybrid+light \
    --embedding-model C:/models/bge-small-en-v1.5 --n-per-class 150 --rpm 10
```

Reports, with 95% CIs: refusal rate on unanswerable questions (1 − this = hallucination rate),
false refusals on questions whose answer *was* in the context, and answer correctness when it
does answer. Every prediction is saved to `predictions_*.jsonl` for manual review. The
`--rpm` limit keeps it inside free-tier rate limits (300 questions at 10 requests per minute is
about 30 minutes).

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
