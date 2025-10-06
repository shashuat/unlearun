# unlearun/evaluation/__init__.py
"""Evaluation module for unlearning."""

from .metrics import (
    evaluate_unlearning,
    compute_perplexity,
    compute_forget_quality,
    compute_model_utility,
    compute_rouge_score,
    compute_verbatim_memorization,
    compute_mia,
    compute_knowledge_retention_qa
)

__all__ = [
    "evaluate_unlearning",
    "compute_perplexity",
    "compute_forget_quality",
    "compute_model_utility",
    "compute_rouge_score",
    "compute_verbatim_memorization",
    "compute_mia",
    "compute_knowledge_retention_qa",
]