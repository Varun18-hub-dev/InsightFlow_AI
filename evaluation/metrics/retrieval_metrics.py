from typing import List, Set
from .base import BaseMetric, MetricResult

class RecallAtK(BaseMetric):
    @property
    def name(self) -> str:
        return "Recall@K"
        
    @property
    def description(self) -> str:
        return "Proportion of relevant documents successfully retrieved in top K."

    def compute(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> MetricResult:
        if not relevant_ids:
            return MetricResult(self.name, 1.0, self.description)
            
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(retrieved_k.intersection(relevant_set))
        
        value = hits / len(relevant_set)
        return MetricResult(self.name, value, self.description, {"k": k})

class PrecisionAtK(BaseMetric):
    @property
    def name(self) -> str:
        return "Precision@K"
        
    @property
    def description(self) -> str:
        return "Proportion of top K retrieved documents that are relevant."

    def compute(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> MetricResult:
        retrieved_k = retrieved_ids[:k]
        if not retrieved_k:
            return MetricResult(self.name, 0.0, self.description, {"k": k})
            
        relevant_set = set(relevant_ids)
        hits = len([doc for doc in retrieved_k if doc in relevant_set])
        
        value = hits / len(retrieved_k)
        return MetricResult(self.name, value, self.description, {"k": k})

class MeanReciprocalRank(BaseMetric):
    @property
    def name(self) -> str:
        return "MRR"
        
    @property
    def description(self) -> str:
        return "Multiplicative inverse of the rank of the first correct answer."

    def compute(self, retrieved_ids: List[str], relevant_ids: List[str]) -> MetricResult:
        if not relevant_ids or not retrieved_ids:
            return MetricResult(self.name, 0.0, self.description)
            
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return MetricResult(self.name, 1.0 / rank, self.description)
                
        return MetricResult(self.name, 0.0, self.description)

class HitRate(BaseMetric):
    @property
    def name(self) -> str:
        return "Hit Rate"
        
    @property
    def description(self) -> str:
        return "Binary metric indicating if at least one relevant document was retrieved."

    def compute(self, retrieved_ids: List[str], relevant_ids: List[str]) -> MetricResult:
        relevant_set = set(relevant_ids)
        hits = any(doc_id in relevant_set for doc_id in retrieved_ids)
        
        return MetricResult(self.name, 1.0 if hits else 0.0, self.description)
