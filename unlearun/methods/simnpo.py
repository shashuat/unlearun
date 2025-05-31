# unlearun/methods/simnpo.py
"""Simple Negative Preference Optimization (SimNPO) unlearning method."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base import BaseUnlearningMethod


class SimNPO(BaseUnlearningMethod):
    """
    Simple Negative Preference Optimization for unlearning.
    
    SimNPO uses a preference-based approach where the model is trained to
    assign lower likelihood to forget examples using a margin-based loss.
    
    Args:
        model: The model to be unlearned
        ref_model: Optional reference model
        delta: Margin parameter for the loss
        beta: Temperature parameter for the loss
        gamma: Weight for forget loss
        alpha: Weight for retain loss
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        delta: float = 0.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 1.0,
        **kwargs
    ):
        super().__init__(model, ref_model, **kwargs)
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
    
    def _compute_batch_nll(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Compute the negative log-likelihood for each sequence in a batch.
        
        Returns:
            loss: Per-sequence loss tensor
            outputs: Model outputs
        """
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift for next-token prediction
        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()
        
        # Compute per-token loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_losses = loss_fct(
            logits.view(-1, logits.size(-1)), 
            shifted_labels.view(-1)
        ).view(labels.size(0), -1)
        
        # Sum over sequence length for each example
        sequence_losses = token_losses.sum(dim=-1)
        
        return sequence_losses, outputs
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Any:
        """
        Compute SimNPO loss.
        
        The forget loss uses a margin-based objective to reduce likelihood
        of forget examples.
        """
        # Prepare forget inputs
        forget_inputs = self.prepare_inputs(inputs["forget"])
        
        # Compute forget loss
        forget_nll, forget_outputs = self._compute_batch_nll(model, forget_inputs)
        
        # Get number of tokens for normalization
        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100
        num_tokens = loss_mask.sum(-1).clamp(min=1)
        
        # Normalize by number of tokens and apply margin
        normalized_forget_loss = forget_nll / num_tokens - self.delta
        
        # Apply logsigmoid for stable optimization
        forget_loss = -F.logsigmoid(self.beta * normalized_forget_loss).mean() * 2 / self.beta
        
        # Prepare retain inputs
        retain_inputs = self.prepare_inputs(inputs["retain"])
        
        # Compute retain loss (standard NLL)
        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss
        
        # Combine losses
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        
        return (loss, forget_outputs) if return_outputs else loss