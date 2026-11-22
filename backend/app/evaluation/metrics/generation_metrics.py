import re

from .base import BaseMetric, MetricResult


class FaithfulnessMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Faithfulness"

    @property
    def description(self) -> str:
        return "Lexical proxy: share of answer words that also appear in the context. Not an LLM or human faithfulness judgement."

    def compute(self, answer: str, context: str) -> MetricResult:
        # Very naive heuristic for demonstration without requiring an LLM call.
        # In a real setup, you'd prompt an LLM to evaluate faithfulness.
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))

        if not answer_words:
            return MetricResult(self.name, 0.0, self.description)

        overlap = len(answer_words.intersection(context_words))
        score = overlap / len(answer_words)

        return MetricResult(self.name, score, self.description)

class AnswerRelevanceMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Answer Relevance"

    @property
    def description(self) -> str:
        return "Lexical proxy: share of question words that also appear in the answer."

    def compute(self, question: str, answer: str) -> MetricResult:
        # Proxy: checks overlap of significant words between question and answer
        q_words = set(re.findall(r'\w+', question.lower()))
        a_words = set(re.findall(r'\w+', answer.lower()))

        if not q_words or not a_words:
            return MetricResult(self.name, 0.0, self.description)

        overlap = len(q_words.intersection(a_words))
        score = overlap / len(q_words)

        return MetricResult(self.name, score, self.description)

class ContextRelevanceMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Context Relevance"

    @property
    def description(self) -> str:
        return "Lexical proxy: share of question words that also appear in the retrieved context."

    def compute(self, question: str, context: str) -> MetricResult:
        q_words = set(re.findall(r'\w+', question.lower()))
        c_words = set(re.findall(r'\w+', context.lower()))

        if not q_words or not c_words:
            return MetricResult(self.name, 0.0, self.description)

        overlap = len(q_words.intersection(c_words))
        score = overlap / len(q_words)

        return MetricResult(self.name, score, self.description)
