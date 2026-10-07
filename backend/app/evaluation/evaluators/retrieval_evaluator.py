from typing import Any

import requests

try:
    from ..metrics.retrieval_metrics import HitRate, MeanReciprocalRank, PrecisionAtK, RecallAtK
except (ImportError, ValueError):
    from metrics.retrieval_metrics import HitRate, MeanReciprocalRank, PrecisionAtK, RecallAtK

class RetrievalEvaluator:
    def __init__(self, api_url: str, token: str = None):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.metrics = [
            RecallAtK(),
            PrecisionAtK(),
            MeanReciprocalRank(),
            HitRate()
        ]

    def evaluate(self, eval_dataset: list[dict[str, Any]]) -> dict[str, float]:
        print(f"Running Retrieval Evaluation on {len(eval_dataset)} samples...")
        results = {metric.name: [] for metric in self.metrics}

        for item in eval_dataset:
            question = item["question"]
            expected_sources = item.get("expected_sources", [])

            # Call backend API
            try:
                response = requests.post(
                    f"{self.api_url}/api/chat",
                    json={"query": question},
                    headers=self.headers,
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    # Extract sources (assuming API returns a list of source dicts with 'document' or 'filename')
                    retrieved_sources = [src.get('document', src.get('filename', src.get('name', ''))) for src in data.get('sources', [])]
                else:
                    retrieved_sources = []
            except Exception as e:
                print(f"Error querying API for '{question}': {e}")
                retrieved_sources = []

            for metric in self.metrics:
                # If metric expects 'k', use default inside metric
                res = metric.compute(retrieved_sources, expected_sources)
                results[metric.name].append(res.value)

        # Aggregate
        aggregated = {name: sum(vals)/len(vals) if vals else 0.0 for name, vals in results.items()}
        return aggregated
