# unlearun/__init__.py# unlearun/__init__.py
"""Unlearun: Machine Unlearning for LLMs."""

__version__ = "0.1.0"

from .core import Unlearning
from .methods import (
    BaseUnlearningMethod,
    GradAscent,
    GradDiff,
    DPO,
    RMU,
    SimNPO,
    get_unlearning_method
)
from .data import (
    UnlearningDataset,
    ForgetRetainDataset,
    DataCollatorForUnlearning
)
from .trainer import UnlearningTrainer
from .evaluation import evaluate_unlearning

__all__ = [
    "Unlearning",
    "BaseUnlearningMethod",
    "GradAscent",
    "GradDiff",
    "DPO",
    "RMU",
    "SimNPO",
    "get_unlearning_method",
    "UnlearningDataset",
    "ForgetRetainDataset",
    "DataCollatorForUnlearning",
    "UnlearningTrainer",
    "evaluate_unlearning",
]