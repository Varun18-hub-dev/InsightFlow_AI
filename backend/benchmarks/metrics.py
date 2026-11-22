"""Per-query retrieval metrics plus bootstrap confidence intervals.

Everything here is a pure function over ranked document ids, so it is unit tested and has
no model or network dependency.
"""
from __future__ import annotations

import math
import re
import string

import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("query has no relevant documents; drop it instead of scoring it")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def normalize_answer(s: str) -> str:
    """Official SQuAD normalisation: lowercase, drop punctuation, articles and extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_in_context(texts: list[str], answers: list[str]) -> float:
    """1.0 if any gold answer string appears in the given passages, after normalisation.

    SQuAD labels only one paragraph as relevant, but the answer often also appears in other
    paragraphs. This metric credits those, so it is reported next to strict recall.
    """
    if not answers:
        return float("nan")
    blob = normalize_answer(" ".join(texts))
    return 1.0 if any(normalize_answer(a) and normalize_answer(a) in blob for a in answers) else 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) confidence interval."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference mean(b) - mean(a) on the same queries, with a 95% CI and a two-sided p-value."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs scores for the same queries")
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"delta": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0))}
