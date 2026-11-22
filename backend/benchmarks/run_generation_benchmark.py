"""End-to-end answer quality on SQuAD: retrieve with InsightFlow's pipeline, answer with the
configured LLM, score with the official SQuAD exact-match and token-F1.

Needs an LLM key (Gemini's free tier is enough for --n-queries 200). Run from backend/:

    GEMINI_API_KEY=... python -m benchmarks.run_generation_benchmark --config bm25+light --n-queries 200

It also reports two things a word-overlap "faithfulness" score cannot:
  - answerable_rate: share of questions whose gold answer was in the retrieved context at all
    (the ceiling for any generator on top of this retriever)
  - abstain_rate:    share of answers where the model said it could not find the answer
Answer F1 is reported separately for questions where the answer WAS retrieved, so retrieval
misses and generation mistakes are not mixed up.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter

from . import datasets, metrics, retrievers
from .run_retrieval_benchmark import HERE, build_configs

ABSTAIN = re.compile(r"(could not|couldn't|cannot|can't|unable to) (find|determine|answer)|not (mentioned|provided|in the)", re.I)

PROMPT = (
    "Answer the question using only the context. Reply with the shortest span that answers it, "
    "with no explanation. If the context does not contain the answer, reply exactly: I could not find the answer.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def squad_f1(pred: str, gold: str) -> float:
    p, g = metrics.normalize_answer(pred).split(), metrics.normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return float(p == g)
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def squad_em(pred: str, gold: str) -> float:
    return float(metrics.normalize_answer(pred) == metrics.normalize_answer(gold))


async def answer_all(llm, items, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        async with sem:
            msg = [{"role": "user", "content": PROMPT.format(context=it["context"], question=it["question"])}]
            try:
                return (await llm.chat(msg)).strip()
            except Exception as e:  # keep going; failures are counted, not hidden
                return f"__ERROR__ {e}"

    return await asyncio.gather(*(one(it) for it in items))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="bm25+light")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--context-chunks", type=int, default=6, help="app default TOP_K_RERANK=6")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from app.llm.factory import LLMProviderFactory

    path = datasets.download_squad_dev(HERE / ".cache" / "squad-dev-v1.1.json")
    corpus, queries = datasets.load_squad(path, n_queries=args.n_queries, seed=args.seed)
    chunks = retrievers.build_chunks(corpus)
    index = build_configs([args.config], chunks, args)[args.config]

    items = []
    for q in queries:
        hits = index.search(q["text"], top_k=20)[: args.context_chunks]
        texts = [h.content for h in hits]
        items.append({"question": q["text"], "answers": q["answers"], "context": "\n\n".join(texts),
                      "answerable": metrics.answer_in_context(texts, q["answers"])})

    llm = LLMProviderFactory.get_provider()
    preds = asyncio.run(answer_all(llm, items, args.concurrency))

    rows = []
    for it, pred in zip(items, preds, strict=True):
        rows.append({
            **{k: it[k] for k in ("question", "answers", "answerable")},
            "prediction": pred,
            "error": pred.startswith("__ERROR__"),
            "abstained": bool(ABSTAIN.search(pred)),
            "em": max(squad_em(pred, g) for g in it["answers"]),
            "f1": max(squad_f1(pred, g) for g in it["answers"]),
        })

    ok = [r for r in rows if not r["error"]]
    ans = [r for r in ok if r["answerable"] == 1.0]
    summary = {
        "config": args.config,
        "model": llm.get_model_name(),
        "n_queries": len(rows),
        "errors": len(rows) - len(ok),
        "answerable_rate": metrics.bootstrap_ci([r["answerable"] for r in ok]),
        "exact_match": metrics.bootstrap_ci([r["em"] for r in ok]),
        "f1": metrics.bootstrap_ci([r["f1"] for r in ok]),
        "f1_when_answer_retrieved": metrics.bootstrap_ci([r["f1"] for r in ans]),
        "abstain_rate": metrics.bootstrap_ci([float(r["abstained"]) for r in ok]),
        "abstain_rate_when_answer_retrieved": metrics.bootstrap_ci([float(r["abstained"]) for r in ans]),
    }
    out = HERE / "results" / "squad_generation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{args.config}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"predictions_{args.config}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
