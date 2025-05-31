# methods/grad_diff.py
"""Gradient Difference unlearning method."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base import BaseUnlearningMethod
from ..utils.losses import compute_kl_divergence


class GradDiff(BaseUnlearningMethod):
    """
    Gradient Difference method for unlearning.
    
    Combines gradient ascent on forget data with gradient descent on retain data.
    Supports KL divergence regularization with a reference model.
    
    Args:
        gamma: Weight for forget loss (default: 1.0)
        alpha: Weight for retain loss (default: 1.0)
        retain_loss_type: Type of retain loss ('NLL' or 'KL', default: 'NLL')
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module] = None,
        gamma: float = 1.0,
        alpha: float = 1.0,
        retain_loss_type: str = "NLL",
        **kwargs
    ):
        super().__init__(model, ref_model, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        
        if self.retain_loss_type not in ["NLL", "KL"]:
            raise ValueError(f"Invalid retain_loss_type: {retain_loss_type}")
    
    @property
    def requires_ref_model(self) -> bool:
        """GradDiff requires ref model only for KL loss."""
        return self.retain_loss_type == "KL"
    
    def compute_retain_loss(
        self,
        model: torch.nn.Module,
        retain_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss on retain set."""
        if self.retain_loss_type == "NLL":
            # Standard negative log-likelihood
            outputs = model(**retain_inputs)
            return outputs.loss
        elif self.retain_loss_type == "KL":
            # KL divergence with reference model
            kl_loss = compute_kl_divergence(model, self.ref_model, retain_inputs)
            return kl_loss
        else:
            raise ValueError(f"Unknown retain loss type: {self.retain_loss_type}")
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Any:
        """
        Compute gradient difference loss.
        
        Loss = gamma * (-forget_loss) + alpha * retain_loss
        """
        # Prepare forget inputs
        forget_inputs = self.prepare_inputs(inputs["forget"])
        
        # Compute forget loss (gradient ascent)
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss
        
        # Prepare retain inputs
        retain_inputs = self.prepare_inputs(inputs["retain"])
        
        # Compute retain loss
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        
        # Combine losses
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        
        return (loss, forget_outputs) if return_outputs else loss