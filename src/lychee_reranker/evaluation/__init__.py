"""Local-data evaluation utilities for Lychee-Reranker-V1."""

from .benchmarks import (
    BEIR_RERANKER_INSTRUCTIONS,
    BEIR_RETRIEVER_INSTRUCTIONS,
    BEIR_V1_TASKS,
    LMEB_DIALOGUE_TASKS,
    MIRACL_LANGUAGES,
)
from .metrics import evaluate_run

__all__ = [
    "BEIR_RERANKER_INSTRUCTIONS",
    "BEIR_RETRIEVER_INSTRUCTIONS",
    "BEIR_V1_TASKS",
    "LMEB_DIALOGUE_TASKS",
    "MIRACL_LANGUAGES",
    "evaluate_run",
]
