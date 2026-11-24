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

### Part 2: does the LLM refuse? (measured)

Run: hybrid + reranker retrieval, `gemini-flash-lite-latest`, the app's production prompts,
150 answerable + 150 unanswerable questions (seed 42), 0 API errors.

```bash
python -m benchmarks.run_abstention_benchmark --configs hybrid+light \
    --embedding-model C:/models/bge-small-en-v1.5 --n-per-class 150 --rpm 10
```

Files: [`summary_hybrid+light.json`](results/squad2_abstention/summary_hybrid+light.json),
[`predictions_hybrid+light.jsonl`](results/squad2_abstention/predictions_hybrid+light.jsonl) (every answer),
[`manual_review_hybrid+light.jsonl`](results/squad2_abstention/manual_review_hybrid+light.jsonl) (my labels).

| Measure | Automatic detector | After manual review |
|---|---|---|
| **Hallucination rate on unanswerable questions** | 28.7% [21.3, 36.0] | **22.7%** [16.0, 29.3] (34 / 150) |
| False refusals (answer was in the context) | 5.5% [2.1, 9.7] | **2.8%** [0.7, 5.5] (4 / 145) |
| Given answers that contain the gold answer | 97.1% | – |
| Refusal precision (when it refuses, it should have) | 93.2% | – |

**Why manual review.** Refusals are detected with a phrase list, so I read every answer the
detector disputed (43 "hallucinations" and 8 "false refusals"). Nine of the 43 were correct:
the model rejected the question's false premise ("the Alps were drained, not flooded"; "the name
was coined in 1777, not 1774") without using a refusal phrase. Two of the 8 "false refusals"
had answered correctly and then added a caveat, and two were refusals of genuinely ambiguous
questions ("When was he elected by Nixon?"). The detector agreed with my labels on 94% of
unanswerable and 97% of answerable questions. All labels are in the manual review file.

**What the 34 real hallucinations have in common.** None look made up from nothing. Each one
restates something the passage says on the same topic, but answers a slightly different
question than the one asked. (The saved predictions do not include the context, so this was
judged from the answers and the SQuAD source paragraphs, not checked word by word.)
- **29 answered a nearby fact without noticing a changed detail.** Asked "Who is the *current*
  Chairman of Obama's Council of Economic Advisors?", it named the *former* chairman. Asked about
  "Warsaw *Palace*", it answered about Warsaw.
- **5 ignored a negation.** Asked "How many Khitan Tumens *weren't* there?", it said there were 3.

**What this means for the fix.** A "quote your evidence" check would probably not catch these,
because the quoted sentence would be real. The fix has to make the model check each detail of the question
(names, dates, numbers, negations) against the context before answering.

### Part 3: candidate fix, A/B tested on the same 300 questions

[`prompts/rag_strict/answer.txt`](prompts/rag_strict/answer.txt) asks the model to check every
detail of the question (names, numbers, dates, negations) against the context, and to reply
"NOT FOUND:" plus the mismatched detail when anything does not match.

Both runs were reviewed by hand with one rubric:
[`manual_review_hybrid+light.jsonl`](results/squad2_abstention/manual_review_hybrid+light.jsonl),
[`manual_review_hybrid+light_rag_strict.jsonl`](results/squad2_abstention/manual_review_hybrid+light_rag_strict.jsonl).
Paired results (same questions, paired bootstrap):
[`comparison_reviewed_app_vs_rag_strict.json`](results/squad2_abstention/comparison_reviewed_app_vs_rag_strict.json).

| Reviewed, same 300 questions | Production prompt | Detail-checking prompt | Paired change |
|---|---|---|---|
| **Hallucination on unanswerable questions** | 22.7% [16.0, 29.3] | **9.3%** [5.3, 14.0] | **−13.3 pts** [−19.3, −8.0], p < 0.001 |
| Refused with no useful answer (answer was in context) | 1.4% (2) | 3.4% (5) | +2.1 pts [0.0, +4.8], p = 0.08 |
| Any "not found" reply when the answer was in context | 2.8% (4) | 7.6% (11) | +4.8 pts [+2.1, +9.0], p = 0.004 |
| Given answers containing the gold answer (automatic) | 97.1% | 94.7% | – |

**What changed.** 26 of the old hallucinations became correct refusals. Examples: "the context
lists Goolsbee as the *former* chairman, not the current one" and "the name oxygen was coined
in 1777, not 1774". One answer that was right before became wrong. 14 hallucinations remain: 11
nearby-fact answers and 3 ignored negations.

**What it costs.** The new prompt is more cautious. Six of its eleven "not found" replies to
answerable questions still state the answer in the same sentence. For example: "NOT FOUND: the
context says such broadcasts are known as free-to-view, not free-to-air." The user gets the
answer, but it is marked as not found. Clear false refusals rose from 2 to 5, which is not
significant (p = 0.08).

**Decision: not promoted to the app yet.** It meets the rule I set (hallucinations down
significantly, real false refusals not up significantly). But the prompt was written *after*
reading the failures on these same 150 unanswerable questions, so this result is optimistic. It
will be promoted only if it holds on a fresh sample, a different `--seed`, with both prompts:

```bash
python -m benchmarks.run_abstention_benchmark --configs hybrid+light --embedding-model C:/models/bge-small-en-v1.5 --n-per-class 150 --rpm 10 --seed 7
python -m benchmarks.run_abstention_benchmark --configs hybrid+light --embedding-model C:/models/bge-small-en-v1.5 --n-per-class 150 --rpm 10 --seed 7 --prompts-dir benchmarks/prompts --prompt-set rag_strict
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
