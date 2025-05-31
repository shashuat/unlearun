# methods/dpo.py
"""Direct Preference Optimization (DPO) unlearning method."""

import torch
from typing import Dict, Any, Optional

from .grad_diff import GradDiff
from ..utils.losses import compute_dpo_loss


class DPO(GradDiff):
    """
    DPO-based unlearning method.
    
    Uses Direct Preference Optimization to unlearn by treating alternate answers
    as preferred over original answers in the forget set.
    
    Args:
        beta: DPO temperature parameter (default: 1.0)
        gamma: Weight for forget loss (default: 1.0)
        alpha: Weight for retain loss (default: 1.0)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module] = None,
        beta: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 1.0,
        **kwargs
    ):
        # DPO always requires a reference model
        super().__init__(
            model=model,
            ref_model=ref_model,
            gamma=gamma,
            alpha=alpha,
            retain_loss_type="NLL",  # DPO uses NLL for retain
            **kwargs
        )
        self.beta = beta
    
    @property
    def requires_ref_model(self) -> bool:
        """DPO always requires a reference model."""
        return True
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Any:
        """
        Compute DPO loss for unlearning.
        
        Expects inputs["forget"] to contain both "original" and "alternate" data.
        """
        # Extract original and alternate answers from forget set
        forget_inputs = inputs["forget"]
        
        # Check if we have alternate answers
        if "alternate" not in forget_inputs:
            # Fall back to standard GradDiff if no alternates
            return super().compute_loss(model, inputs, return_outputs)
        
        original_inputs = self.prepare_inputs(forget_inputs["original"])
        alternate_inputs = self.prepare_inputs(forget_inputs["alternate"])
        
        # Compute DPO loss (alternate is preferred over original)
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=alternate_inputs,  # Alternate answers are "winning"
            lose_inputs=original_inputs,  # Original answers are "losing"
            beta=self.beta
        )
        
        # Compute retain loss
        retain_inputs = self.prepare_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        
        # Combine losses
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        
        return (loss, forget_outputs) if return_outputs else loss