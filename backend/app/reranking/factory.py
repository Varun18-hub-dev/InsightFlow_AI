from app.core.config import settings

from .base_reranker import BaseReranker
from .cross_encoder_reranker import CrossEncoderReranker
from .lightweight_reranker import LightweightReranker


def get_reranker() -> BaseReranker:
    if settings.RERANKER_TYPE == "cross_encoder":
        return CrossEncoderReranker()
    return LightweightReranker()
