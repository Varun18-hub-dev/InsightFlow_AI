"""Paired comparison of two abstention runs on the same questions (e.g. app prompt vs candidate).

    python -m benchmarks.compare_abstention \\
        benchmarks/results/squad2_abstention/predictions_hybrid+light.jsonl \\
        benchmarks/results/squad2_abstention/predictions_hybrid+light_rag_strict.jsonl

Uses the automatic refusal detector for both runs, so the comparison is like for like. Reports
the change in hallucination rate on unanswerable questions and in false refusals on questions
whose answer was retrieved, each with a paired bootstrap CI, plus which questions flipped.
"""
from __future__ import annotations

import json
import sys

from . import metrics


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f) if not r.get("error")}


def main(a_path: str, b_path: str):
    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    un = [q for q in common if a[q]["unanswerable"]]
    ok = [q for q in common if a[q]["answer_in_context"]]
    report = {
        "n_common": len(common),
        "hallucination_unanswerable": {
            "a": round(sum(not a[q]["refused"] for q in un) / len(un), 4),
            "b": round(sum(not b[q]["refused"] for q in un) / len(un), 4),
            "paired": metrics.paired_bootstrap([float(not a[q]["refused"]) for q in un],
                                               [float(not b[q]["refused"]) for q in un]),
        },
        "false_refusal": {
            "a": round(sum(a[q]["refused"] for q in ok) / len(ok), 4),
            "b": round(sum(b[q]["refused"] for q in ok) / len(ok), 4),
            "paired": metrics.paired_bootstrap([float(a[q]["refused"]) for q in ok],
                                               [float(b[q]["refused"]) for q in ok]),
        },
        "contains_gold_when_answered": {
            "a": round(sum(a[q]["contains_gold"] for q in ok if not a[q]["refused"])
                       / max(1, sum(not a[q]["refused"] for q in ok)), 4),
            "b": round(sum(b[q]["contains_gold"] for q in ok if not b[q]["refused"])
                       / max(1, sum(not b[q]["refused"] for q in ok)), 4),
        },
        "fixed_hallucinations": [a[q]["question"] for q in un if not a[q]["refused"] and b[q]["refused"]],
        "new_hallucinations": [a[q]["question"] for q in un if a[q]["refused"] and not b[q]["refused"]],
        "new_false_refusals": [a[q]["question"] for q in ok if not a[q]["refused"] and b[q]["refused"]],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
