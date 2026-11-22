# Legacy evaluation results (do not cite)

These two files are the project's first evaluation runs (October 2026). They are kept as a
record, not as evidence of quality.

| File | What it measured | Why it is not meaningful |
|---|---|---|
| `evaluation_results.json` | 15 hand-written questions | Every retrieval score is 1.0, so it cannot tell configurations apart. Faithfulness and relevance were word overlap multiplied by 1.5 and capped at 1.0. |
| `evaluation_benchmark_pinecone_vs_inmemory.json` | 10 questions over a 5-document corpus | With 5 documents almost any retriever finds the right one. The latency numbers are real for that tiny index, but they say nothing about scale. |

What replaced them: [`backend/benchmarks/`](../../backend/benchmarks/README.md), a public dataset
(SQuAD v1.1 dev, 2,067 queries) with confidence intervals and error analysis.
