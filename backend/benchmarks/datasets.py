"""Loaders for public retrieval benchmarks.

Every loader returns the same shape so the runner does not care where data came from:

    corpus:  {doc_id: {"title": str, "text": str, "group": str}}
    queries: [{"qid": str, "text": str, "relevant": set[str], "answers": list[str], "group": str}]

`group` is the article / topic a document belongs to. It lets the error analysis tell a
near miss (right article, wrong paragraph) from a real miss.
"""
from __future__ import annotations

import csv
import json
import random
import urllib.request
from pathlib import Path

SQUAD_DEV_URL = (
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
)


def download_squad_dev(dest: Path) -> Path:
    """Download the official SQuAD v1.1 dev set (about 4.8 MB) if it is not cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(SQUAD_DEV_URL, dest)  # noqa: S310 - fixed https URL
    return dest


def load_squad(path: Path, n_queries: int | None = 1000, seed: int = 42) -> tuple[dict, list]:
    """SQuAD v1.1 dev as a retrieval task.

    Corpus = every paragraph in the dev set (2,067 paragraphs from 48 Wikipedia articles).
    Query  = a question; its only labelled relevant document is the paragraph it was written on.

    Questions are sampled with a fixed seed, at most one per paragraph, so no paragraph
    dominates the score.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    corpus: dict[str, dict] = {}
    per_paragraph: list[list[dict]] = []
    for a_idx, article in enumerate(data):
        title = article["title"].replace("_", " ")
        for p_idx, para in enumerate(article["paragraphs"]):
            doc_id = f"squad-{a_idx}-{p_idx}"
            corpus[doc_id] = {"title": title, "text": para["context"], "group": title}
            per_paragraph.append([
                {
                    "qid": qa["id"],
                    "text": qa["question"].strip(),
                    "relevant": {doc_id},
                    "answers": sorted({a["text"] for a in qa["answers"]}),
                    "group": title,
                }
                for qa in para["qas"]
            ])

    rng = random.Random(seed)
    queries = [rng.choice(qs) for qs in per_paragraph if qs]
    rng.shuffle(queries)
    if n_queries is not None:
        queries = queries[:n_queries]
    return corpus, queries


def load_beir(folder: Path, split: str = "test", n_queries: int | None = None, seed: int = 42) -> tuple[dict, list]:
    """Any dataset in BEIR format (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

    Use this for SciFact, FiQA, NFCorpus etc. after downloading them on a machine that can
    reach the BEIR mirrors. BEIR has no answer strings, so answer-in-context is skipped.
    """
    folder = Path(folder)
    corpus = {}
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "group": str(row["_id"]),
            }
    qtext = {}
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qtext[str(row["_id"])] = row["text"]
    rel: dict[str, set[str]] = {}
    with open(folder / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for qid, did, score in reader:
            if int(score) > 0:
                rel.setdefault(qid, set()).add(did)
    queries = [
        {"qid": q, "text": qtext[q], "relevant": docs, "answers": [], "group": ""}
        for q, docs in sorted(rel.items())
        if q in qtext
    ]
    random.Random(seed).shuffle(queries)
    if n_queries is not None:
        queries = queries[:n_queries]
    return corpus, queries
