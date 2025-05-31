# methods/grad_ascent.py
"""Gradient Ascent unlearning method."""

import torch
from typing import Dict, Any

from .base import BaseUnlearningMethod


class GradAscent(BaseUnlearningMethod):
    """
    Gradient Ascent method for unlearning.
    
    This method maximizes the loss on the forget set by using negative loss.
    It's the simplest unlearning approach but can be unstable.
    """
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Any:
        """
        Compute gradient ascent loss.
        
        Simply negates the loss on the forget set to maximize it.
        """
        forget_inputs = inputs["forget"]
        forget_inputs = self.prepare_inputs(forget_inputs)
        
        # Forward pass on forget data
        outputs = model(**forget_inputs)
        
        # Negate loss to perform gradient ascent
        loss = -outputs.loss
        
        return (loss, outputs) if return_outputs else loss