"""Does InsightFlow say "I don't know" when the documents do not contain the answer?

Uses SQuAD 2.0 dev, which adds about 6,000 *unanswerable* questions to SQuAD. Each one was
written by crowdworkers to look answerable from a specific paragraph (same topic, same words),
but the paragraph does not actually contain the answer. That makes them a hard test of
whether a RAG system refuses or makes something up.

The pipeline is the app's own: same chunker, same retrieval/reranking stages, top 6 chunks,
the same "[Source: file, Page n]" context format as app/api/chat.py, and the production
prompts from prompts/rag/system.txt and prompts/rag/answer.txt.

Two parts:

1. Retrieval signal (no LLM, no key): can the top retrieval score alone tell answerable from
   unanswerable questions? Reported as AUROC (0.5 = no better than chance).
       python -m benchmarks.run_abstention_benchmark --configs bm25+light --retrieval-only

2. End to end (needs an LLM key; Gemini free tier works):
       python -m benchmarks.run_abstention_benchmark --configs hybrid+light \
           --embedding-model C:/models/bge-small-en-v1.5 --n-per-class 150 --rpm 10

Scoring. A question is *should-refuse* if it is unanswerable, or if it is answerable but the
gold answer is not in the 6 chunks given to the model (then refusing is the right behaviour
for a grounded assistant). Reported:
  - refusal_rate_unanswerable   higher is better; 1 - this = hallucination rate
  - false_refusal_rate          refusals on questions whose answer WAS in the context; lower is better
  - contains_gold_when_answered  share of given answers that contain a gold answer string
  - f1_when_answered            SQuAD token F1 on those answers (lowered by the app's verbose, cited style)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import metrics, retrievers
from .run_generation_benchmark import squad_em, squad_f1
from .run_retrieval_benchmark import HERE, build_configs

SQUAD2_DEV_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json"

# Phrases from prompts/rag/system.txt and answer.txt, plus common paraphrases.
REFUSAL = re.compile(
    r"could not be found in the provided documents|could not find sufficient information"
    r"|(could not|couldn't|cannot|can't|unable to|do not|don't) (find|locate|determine|answer)"
    r"|(does not|doesn't|do not|don't) (contain|mention|provide|specify|include|state)"
    r"|(no|not enough|insufficient) (information|mention|details?|evidence)"
    r"|not (mentioned|provided|specified|stated|available) in",
    re.I,
)
CITATION = re.compile(r"\(\s*Source:[^)]*\)", re.I)


def is_refusal(text: str) -> bool:
    # "NOT FOUND:" is the fixed marker the candidate prompt in benchmarks/prompts/rag_strict asks for.
    return text.lstrip().upper().startswith("NOT FOUND") or bool(REFUSAL.search(text))


def strip_citations(text: str) -> str:
    return CITATION.sub("", text).strip()


def load_squad2(path: Path, n_per_class: int | None, seed: int) -> tuple[dict, list]:
    """Corpus = every SQuAD 2.0 dev paragraph. Queries = up to one answerable and one
    unanswerable question per paragraph, then n_per_class of each, balanced."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    rng = random.Random(seed)
    corpus, ans, unans = {}, [], []
    for a_idx, art in enumerate(data):
        title = art["title"].replace("_", " ")
        for p_idx, para in enumerate(art["paragraphs"]):
            doc_id = f"squad2-{a_idx}-{p_idx}"
            corpus[doc_id] = {"title": title, "text": para["context"], "group": title}
            pools = {True: [], False: []}
            for qa in para["qas"]:
                pools[bool(qa.get("is_impossible"))].append(qa)
            for impossible, bucket in ((False, ans), (True, unans)):
                if pools[impossible]:
                    qa = rng.choice(pools[impossible])
                    bucket.append({
                        "qid": qa["id"], "text": qa["question"].strip(), "relevant": {doc_id},
                        "answers": sorted({a["text"] for a in qa["answers"]}),
                        "unanswerable": impossible, "group": title,
                    })
    rng.shuffle(ans)
    rng.shuffle(unans)
    if n_per_class:
        ans, unans = ans[:n_per_class], unans[:n_per_class]
    queries = ans + unans
    rng.shuffle(queries)
    return corpus, queries


def auroc(pos: list[float], neg: list[float]) -> float:
    """Probability that a random positive scores higher than a random negative (ties = 0.5)."""
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def format_context(hits, corpus) -> str:
    """Same layout as app/api/chat.py."""
    parts = [f"[Source: {corpus[h.metadata['doc_id']]['title']}, Page 1]\n{h.content}" for h in hits]
    return "\n\n---\n\n".join(parts) if parts else "No context available."


QUOTA = re.compile(r"429|RESOURCE_EXHAUSTED|quota|rate limit", re.I)


class QuotaExhausted(RuntimeError):
    pass


async def ask_all(llm, prompts: list[tuple[str, str]], keys: list[str], concurrency: int, rpm: float,
                  checkpoint: Path, timeout: float = 120.0):
    """Ask every prompt not already answered in `checkpoint`, saving each answer as it arrives.

    A crash, a closed laptop or a used-up daily quota loses nothing: rerun the same command and
    it continues from the checkpoint. Errors are not saved, so they are retried on the next run.
    After 5 quota errors in a row it stops instead of hammering the API.
    """
    done: dict[str, str] = {}
    if checkpoint.exists():
        with open(checkpoint, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["qid"]] = row["prediction"]
    todo = [i for i, k in enumerate(keys) if k not in done]
    print(f"{len(done)} answers already saved in {checkpoint.name}; asking {len(todo)} more")

    sem = asyncio.Semaphore(concurrency)
    gap = 60.0 / rpm if rpm else 0.0
    lock, last, quota_errors = asyncio.Lock(), [0.0], [0]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    out = open(checkpoint, "a", encoding="utf-8")

    async def one(i):
        system, user = prompts[i]
        async with sem:
            for attempt in range(3):
                if quota_errors[0] >= 5:
                    return
                if gap:
                    async with lock:
                        wait = last[0] + gap - time.monotonic()
                        if wait > 0:
                            await asyncio.sleep(wait)
                        last[0] = time.monotonic()
                try:
                    pred = (await asyncio.wait_for(
                        llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}]),
                        timeout)).strip()
                    quota_errors[0] = 0
                    done[keys[i]] = pred
                    out.write(json.dumps({"qid": keys[i], "prediction": pred}) + "\n")
                    out.flush()
                    return
                except Exception as e:  # timeouts, rate limits; never saved, so retried next run
                    if QUOTA.search(str(e)):
                        quota_errors[0] += 1
                    if attempt < 2:
                        await asyncio.sleep(20 * (attempt + 1))

    try:
        await asyncio.gather(*(one(i) for i in todo))
    finally:
        out.close()
    missing = [k for k in keys if k not in done]
    if missing:
        hint = ("The API keeps returning quota/rate-limit errors, most likely the free tier's daily limit. "
                if quota_errors[0] >= 5 else "")
        raise QuotaExhausted(f"{len(missing)} of {len(keys)} questions still unanswered. {hint}"
                             f"Progress is saved; rerun the same command later to continue.")
    return [done[k] for k in keys]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default="bm25+light", help="one retrieval config, e.g. hybrid+light")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--n-per-class", type=int, default=150, help="answerable and unanswerable each; 0 = all")
    p.add_argument("--context-chunks", type=int, default=6, help="app default TOP_K_RERANK=6")
    p.add_argument("--prompts-dir", default=None,
                   help="folder of prompt sets; default is the app's own prompts/ folder")
    p.add_argument("--prompt-set", default="rag",
                   help="prompt set inside --prompts-dir: 'rag' = the app's production prompt; "
                        "'rag_strict' (with --prompts-dir benchmarks/prompts) = the detail-checking candidate")
    p.add_argument("--retrieval-only", action="store_true", help="skip the LLM; report the retrieval-score AUROC only")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--rpm", type=float, default=10, help="max LLM requests per minute (free tiers are low)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(HERE / "results"))
    args = p.parse_args()
    config = args.configs.split(",")[0]

    path = HERE / ".cache" / "squad-dev-v2.0.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD2_DEV_URL, path)  # noqa: S310 - fixed https URL
    corpus, queries = load_squad2(path, args.n_per_class or None, args.seed)
    chunks = retrievers.build_chunks(corpus)
    index = build_configs([config], chunks, args)[config]
    print(f"corpus: {len(corpus)} paragraphs -> {len(chunks)} chunks | "
          f"{sum(not q['unanswerable'] for q in queries)} answerable + {sum(q['unanswerable'] for q in queries)} unanswerable")

    rows = []
    for q in queries:
        hits = index.search(q["text"], top_k=20)[: args.context_chunks]
        texts = [h.content for h in hits]
        rows.append({
            "qid": q["qid"], "question": q["text"], "article": q["group"], "unanswerable": q["unanswerable"],
            "answers": q["answers"],
            "answer_in_context": (not q["unanswerable"]) and metrics.answer_in_context(texts, q["answers"]) == 1.0,
            "gold_paragraph_in_context": any(h.metadata["doc_id"] in q["relevant"] for h in hits),
            "top_score": hits[0].score if hits else 0.0,
            "context": format_context(hits, corpus),
        })
    for r in rows:
        r["should_refuse"] = r["unanswerable"] or not r["answer_in_context"]

    summary = {
        "dataset": "squad2-dev",
        "config": config,
        "prompt_set": args.prompt_set,
        "n_answerable": sum(not r["unanswerable"] for r in rows),
        "n_unanswerable": sum(r["unanswerable"] for r in rows),
        "seed": args.seed,
        "retrieval_signal": {
            "auroc_top_score_answerable_vs_unanswerable": round(auroc(
                [r["top_score"] for r in rows if not r["unanswerable"]],
                [r["top_score"] for r in rows if r["unanswerable"]]), 4),
            "gold_paragraph_in_context_unanswerable": round(
                float(np.mean([r["gold_paragraph_in_context"] for r in rows if r["unanswerable"]])), 4),
            "answer_in_context_answerable": round(
                float(np.mean([r["answer_in_context"] for r in rows if not r["unanswerable"]])), 4),
        },
    }

    if not args.retrieval_only:
        from app.core.prompt_manager import PromptManager
        from app.llm.factory import LLMProviderFactory

        pm = PromptManager(Path(args.prompts_dir) if args.prompts_dir else None)
        system = pm.get_prompt(args.prompt_set, "system")
        if not system or not pm.get_prompt(args.prompt_set, "answer"):
            raise SystemExit(f"prompt set '{args.prompt_set}' not found in {pm.base_dir}")
        prompts = [(system, pm.render(args.prompt_set, "answer", context=r["context"], question=r["question"]))
                   for r in rows]
        llm = LLMProviderFactory.get_provider()
        print(f"asking {llm.get_model_name()} {len(prompts)} questions at <= {args.rpm} requests/min ...")
        ckpt = HERE / ".cache" / f"abstention_{config}_{args.prompt_set}_seed{args.seed}_n{args.n_per_class}.jsonl"
        try:
            preds = asyncio.run(ask_all(llm, prompts, [r["qid"] for r in rows], args.concurrency, args.rpm, ckpt))
        except QuotaExhausted as e:
            raise SystemExit(str(e)) from None
        for r, pred in zip(rows, preds, strict=True):
            r["prediction"] = pred
            r["error"] = pred.startswith("__ERROR__")
            r["refused"] = is_refusal(pred)
            clean = strip_citations(pred)
            r["f1"] = max((squad_f1(clean, g) for g in r["answers"]), default=0.0)
            r["em"] = max((squad_em(clean, g) for g in r["answers"]), default=0.0)
            # The app prompt asks for full sentences with citations, which lowers token F1 against
            # short gold spans. Containment is robust to that verbosity.
            r["contains_gold"] = metrics.answer_in_context([clean], r["answers"]) == 1.0 if r["answers"] else False

        ok = [r for r in rows if not r["error"]]
        un = [r for r in ok if r["unanswerable"]]
        in_ctx = [r for r in ok if r["answer_in_context"]]
        miss_ctx = [r for r in ok if not r["unanswerable"] and not r["answer_in_context"]]
        answered = [r for r in in_ctx if not r["refused"]]
        should = [r for r in ok if r["should_refuse"]]

        def ci(vals):
            m, lo, hi = metrics.bootstrap_ci([float(v) for v in vals])
            return {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}

        tp = sum(r["refused"] for r in should)
        fp = sum(r["refused"] for r in ok if not r["should_refuse"])
        summary["llm"] = {
            "model": llm.get_model_name(),
            "errors": len(rows) - len(ok),
            "refusal_rate_unanswerable": ci([r["refused"] for r in un]),
            "hallucination_rate_unanswerable": ci([not r["refused"] for r in un]),
            "false_refusal_rate": ci([r["refused"] for r in in_ctx]),
            "refusal_rate_answer_not_retrieved": ci([r["refused"] for r in miss_ctx]),
            "f1_when_answered": ci([r["f1"] for r in answered]),
            "contains_gold_when_answered": ci([r["contains_gold"] for r in answered]),
            "exact_match_when_answered": ci([r["em"] for r in answered]),
            "refusal_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "refusal_recall": round(tp / len(should), 4) if should else None,
        }

    out = Path(args.out) / "squad2_abstention"
    out.mkdir(parents=True, exist_ok=True)
    tag = config + ("_retrieval" if args.retrieval_only else "") + (
        "" if args.prompt_set == "rag" or args.retrieval_only else f"_{args.prompt_set}")
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / f"predictions_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "context"}) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
