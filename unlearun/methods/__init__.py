# methods/__init__.py
"""Unlearning methods."""

from .base import BaseUnlearningMethod
from .grad_ascent import GradAscent
from .grad_diff import GradDiff
from .dpo import DPO
from .rmu import RMU
from .simnpo import SimNPO

__all__ = [
    "BaseUnlearningMethod",
    "GradAscent",
    "GradDiff",
    "DPO",
    "RMU",
    "SimNPO",
]

# Registry for method lookup
METHOD_REGISTRY = {
    "GradAscent": GradAscent,
    "GradDiff": GradDiff,
    "DPO": DPO,
    "RMU": RMU,
    "SimNPO": SimNPO,
}


def get_unlearning_method(method_name: str):
    """Get unlearning method class by name."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available methods: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[method_name]