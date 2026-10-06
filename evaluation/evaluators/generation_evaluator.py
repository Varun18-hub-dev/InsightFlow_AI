import requests
import time
from typing import Dict, List, Any
try:
    from ..metrics.generation_metrics import FaithfulnessMetric, AnswerRelevanceMetric, ContextRelevanceMetric
except (ImportError, ValueError):
    from metrics.generation_metrics import FaithfulnessMetric, AnswerRelevanceMetric, ContextRelevanceMetric

class GenerationEvaluator:
    def __init__(self, api_url: str, token: str = None):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.metrics = [
            FaithfulnessMetric(),
            AnswerRelevanceMetric(),
            ContextRelevanceMetric()
        ]

    def evaluate(self, eval_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        print(f"Running Generation Evaluation on {len(eval_dataset)} samples...")
        results = {metric.name: [] for metric in self.metrics}
        results["Latency (s)"] = []
        
        for item in eval_dataset:
            question = item["question"]
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_url}/api/chat",
                    json={"query": question},
                    headers=self.headers,
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    # Concatenate source text if available
                    context = " ".join([src.get("content", "") for src in data.get("sources", [])])
                else:
                    answer = ""
                    context = ""
            except Exception as e:
                print(f"Error querying API for '{question}': {e}")
                answer = ""
                context = ""
                
            latency = time.time() - start_time
            results["Latency (s)"].append(latency)

            for metric in self.metrics:
                if metric.name == "Faithfulness":
                    val = metric.compute(answer, context).value
                elif metric.name == "Answer Relevance":
                    val = metric.compute(question, answer).value
                elif metric.name == "Context Relevance":
                    val = metric.compute(question, context).value
                results[metric.name].append(val)
                
        # Aggregate
        aggregated = {name: sum(vals)/len(vals) if vals else 0.0 for name, vals in results.items()}
        return aggregated
