"""Retrieval benchmark for InsightFlow AI.

Compares the app's retrieval configurations on a public dataset, with confidence intervals,
a paired significance test against the BM25 baseline, and an error analysis.

Examples (run from backend/):

    # Keyword-only configs, CPU only, no model download
    python -m benchmarks.run_retrieval_benchmark --configs bm25,bm25+light

    # Full comparison with a local embedding model (path or Hugging Face name)
    python -m benchmarks.run_retrieval_benchmark \
        --configs bm25,dense,hybrid,hybrid+light,hybrid+ce \
        --embedding-model BAAI/bge-small-en-v1.5

    # Any BEIR dataset you downloaded (e.g. SciFact)
    python -m benchmarks.run_retrieval_benchmark --dataset beir --beir-dir data/scifact
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from . import datasets, metrics, retrievers

KS = (1, 5, 10)
HERE = Path(__file__).resolve().parent


def build_configs(names: list[str], chunks, args) -> dict:
    from app.reranking.cross_encoder_reranker import CrossEncoderReranker, _load_cross_encoder
    from app.reranking.lightweight_reranker import LightweightReranker

    needs_dense = any(n.startswith(("dense", "hybrid")) for n in names)
    bm25 = retrievers.BM25Index(chunks)
    dense = (
        retrievers.DenseIndex(chunks, args.embedding_model, cache_dir=HERE / ".cache") if needs_dense else None
    )
    base = {"bm25": bm25, "dense": dense}
    if dense is not None:
        base["hybrid"] = retrievers.HybridIndex(bm25, dense)

    out = {}
    for name in names:
        stem, _, rr = name.partition("+")
        if stem not in base or base[stem] is None:
            raise SystemExit(f"unknown or unavailable config: {name}")
        if not rr:
            out[name] = base[stem]
        elif rr == "light":
            out[name] = retrievers.Reranked(base[stem], LightweightReranker(), name)
        elif rr == "ce":
            if _load_cross_encoder() is None:
                # The app silently falls back to the lightweight reranker. A benchmark must not.
                raise SystemExit("cross-encoder model could not be loaded; refusing to mislabel results")
            out[name] = retrievers.Reranked(base[stem], CrossEncoderReranker(), name)
        else:
            raise SystemExit(f"unknown reranker suffix in {name}")
    return out


def classify(rec: dict, corpus: dict) -> str:
    rank = rec["gold_rank"]
    if rank == 1:
        return "hit@1"
    if rank is not None and rank <= 5:
        return "hit@2-5"
    if rec.get("answer_in_top5") == 1.0:
        return "answer_in_other_passage"
    top1 = rec["top_docs"][0] if rec["top_docs"] else None
    if top1 and corpus[top1]["group"] == rec["group"]:
        return "same_article_wrong_paragraph"
    return "miss"


def run(args) -> Path:
    if args.dataset == "squad":
        path = datasets.download_squad_dev(HERE / ".cache" / "squad-dev-v1.1.json")
        corpus, queries = datasets.load_squad(path, n_queries=args.n_queries or None, seed=args.seed)
    else:
        corpus, queries = datasets.load_beir(Path(args.beir_dir), n_queries=args.n_queries or None, seed=args.seed)

    chunks = retrievers.build_chunks(corpus)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks | queries: {len(queries)}")
    configs = build_configs(args.configs.split(","), chunks, args)

    per_query: dict[str, list[dict]] = {}
    for name, index in configs.items():
        recs, lat = [], []
        for q in queries:
            t0 = time.perf_counter()
            hits = index.search(q["text"], top_k=args.top_k)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = retrievers.collapse_to_docs(hits)
            gold_rank = next((i for i, d in enumerate(ranked, 1) if d in q["relevant"]), None)
            rec = {
                "qid": q["qid"],
                "query": q["text"],
                "group": q["group"],
                "gold": sorted(q["relevant"]),
                "gold_rank": gold_rank,
                "top_docs": ranked[:5],
                "mrr@10": metrics.mrr_at_k(ranked, q["relevant"], 10),
                "ndcg@10": metrics.ndcg_at_k(ranked, q["relevant"], 10),
            }
            for k in KS:
                rec[f"recall@{k}"] = metrics.recall_at_k(ranked, q["relevant"], k)
            if q["answers"]:
                top5_text = [h.content for h in hits if h.metadata["doc_id"] in set(ranked[:5])]
                rec["answer_in_top5"] = metrics.answer_in_context(top5_text, q["answers"])
            rec["bucket"] = classify(rec, corpus)
            recs.append(rec)
        per_query[name] = recs
        per_query[name + "__latency"] = lat
        print(f"  {name:14s} recall@5={statistics.mean(r['recall@5'] for r in recs):.3f}")

    names = list(configs)
    metric_keys = [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10"]
    if queries[0]["answers"]:
        metric_keys.append("answer_in_top5")

    summary = {
        "dataset": args.dataset,
        "n_docs": len(corpus),
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "seed": args.seed,
        "embedding_model": args.embedding_model if any("dense" in n or "hybrid" in n for n in names) else None,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "configs": {},
        "vs_baseline": {},
        "error_buckets": {},
    }
    baseline = names[0]
    for name in names:
        recs = per_query[name]
        lat = sorted(per_query[name + "__latency"])
        entry = {}
        for m in metric_keys:
            mean, lo, hi = metrics.bootstrap_ci([r[m] for r in recs])
            entry[m] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        entry["latency_ms_p50"] = round(lat[len(lat) // 2], 2)
        entry["latency_ms_p95"] = round(lat[int(len(lat) * 0.95) - 1], 2)
        summary["configs"][name] = entry
        buckets: dict[str, int] = {}
        for r in recs:
            buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
        summary["error_buckets"][name] = buckets
        if name != baseline:
            summary["vs_baseline"][name] = {
                m: metrics.paired_bootstrap([r[m] for r in per_query[baseline]], [r[m] for r in recs])
                for m in ("recall@5", "mrr@10")
            }

    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
        for name in names:
            for r in per_query[name]:
                f.write(json.dumps({"config": name, **r}) + "\n")
    (out / "summary.md").write_text(render_markdown(summary, per_query, corpus, metric_keys))
    print(f"wrote {out}")
    return out


def render_markdown(summary: dict, per_query: dict, corpus: dict, metric_keys: list[str]) -> str:
    names = list(summary["configs"])
    lines = [
        f"# Retrieval benchmark: {summary['dataset']}",
        "",
        f"{summary['n_queries']} queries over {summary['n_docs']} documents ({summary['n_chunks']} chunks). "
        f"Seed {summary['seed']}. Run {summary['run_at']} on {summary['machine']}.",
        "",
        "Means with 95% bootstrap confidence intervals.",
        "",
        "| Config | " + " | ".join(metric_keys) + " | p50 ms | p95 ms |",
        "|---" * (len(metric_keys) + 3) + "|",
    ]
    for n in names:
        e = summary["configs"][n]
        cells = [f"{e[m]['mean']:.3f} [{e[m]['ci95'][0]:.3f}, {e[m]['ci95'][1]:.3f}]" for m in metric_keys]
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {e['latency_ms_p50']} | {e['latency_ms_p95']} |")
    if summary["vs_baseline"]:
        lines += ["", f"## Paired difference vs `{names[0]}` (same queries)", "",
                  "| Config | Δ recall@5 [95% CI] | p | Δ MRR@10 [95% CI] | p |", "|---|---|---|---|---|"]
        for n, d in summary["vs_baseline"].items():
            r, m = d["recall@5"], d["mrr@10"]
            lines.append(
                f"| {n} | {r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_value']:.3f} "
                f"| {m['delta']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}] | {m['p_value']:.3f} |"
            )
    lines += ["", "## Where each config fails", "",
              "| Config | " + " | ".join(BUCKETS) + " |", "|---" * (len(BUCKETS) + 1) + "|"]
    for n in names:
        b = summary["error_buckets"][n]
        lines.append(f"| {n} | " + " | ".join(str(b.get(k, 0)) for k in BUCKETS) + " |")
    lines += ["", "Bucket meanings: `answer_in_other_passage` = the labelled paragraph was missed but a "
              "top-5 passage still contains the gold answer (often a labelling gap, not a retrieval error); "
              "`same_article_wrong_paragraph` = the top result is from the right article; `miss` = neither.", ""]
    worst = names[-1]
    misses = [r for r in per_query[worst] if r["bucket"] == "miss"][:10]
    if misses:
        lines += [f"## Sample misses for `{worst}`", ""]
        for r in misses:
            got = corpus[r["top_docs"][0]]["title"] if r["top_docs"] else "-"
            lines.append(f"- **{r['query']}** (article: {r['group']}) -> top result from: {got}")
    return "\n".join(lines) + "\n"


BUCKETS = ["hit@1", "hit@2-5", "answer_in_other_passage", "same_article_wrong_paragraph", "miss"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["squad", "beir"], default="squad")
    p.add_argument("--beir-dir", help="folder with corpus.jsonl, queries.jsonl, qrels/test.tsv")
    p.add_argument("--configs", default="bm25,bm25+light",
                   help="comma list from: bm25, dense, hybrid, each optionally +light or +ce")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5",
                   help="sentence-transformers name or local folder")
    p.add_argument("--n-queries", type=int, default=0, help="0 = all (one question per paragraph)")
    p.add_argument("--top-k", type=int, default=20, help="candidates per query (app default TOP_K_RETRIEVAL=20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(HERE / "results"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
