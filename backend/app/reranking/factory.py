from app.core.config import settings

from .base_reranker import BaseReranker
from .cross_encoder_reranker import CrossEncoderReranker
from .lightweight_reranker import LightweightReranker


def get_reranker(reranker_type: str | None = None) -> BaseReranker:
    rtype = reranker_type or settings.RERANKER_TYPE
    if rtype == "cross_encoder":
        return CrossEncoderReranker()
    return LightweightReranker()
