# unlearun/utils/losses.py
"""Loss functions for unlearning methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


def compute_kl_divergence(
    model: nn.Module,
    ref_model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute KL divergence between model and reference model outputs.
    
    Args:
        model: Current model
        ref_model: Reference model
        inputs: Input dictionary
        temperature: Temperature for softmax (higher = softer distribution)
    
    Returns:
        KL divergence loss
    """
    # Get reference model outputs (no gradients needed)
    with torch.no_grad():
        ref_outputs = ref_model(**inputs)
        ref_logits = ref_outputs.logits
    
    # Get current model outputs
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()
    
    # Apply temperature
    shift_logits = shift_logits / temperature
    shift_ref_logits = shift_ref_logits / temperature
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    ref_probs = F.softmax(shift_ref_logits, dim=-1)
    
    # Compute KL divergence: KL(ref || model)
    # KL = sum(ref_probs * (log(ref_probs) - log(model_probs)))
    kl_div = F.kl_div(
        log_probs.view(-1, log_probs.size(-1)),
        ref_probs.view(-1, ref_probs.size(-1)),
        reduction='none',
        log_target=False
    ).sum(-1)
    
    # Mask out padded tokens
    loss_mask = (shift_labels != -100).view(-1)
    kl_div = kl_div * loss_mask
    
    # Average over non-masked tokens
    # Clamp to 0 to handle numerical precision issues (KL divergence should be non-negative)
    return (kl_div.sum() / loss_mask.sum().clamp(min=1)).clamp(min=0.0)


def compute_dpo_loss(
    model: nn.Module,
    ref_model: nn.Module,
    win_inputs: Dict[str, torch.Tensor],
    lose_inputs: Dict[str, torch.Tensor],
    beta: float = 1.0
) -> Tuple[torch.Tensor, Any]:
    """
    Compute Direct Preference Optimization (DPO) loss.
    
    For unlearning:
    - win_inputs: alternate/preferred answers (what we want)
    - lose_inputs: original/forget answers (what we don't want)
    
    Args:
        model: Current model
        ref_model: Reference model
        win_inputs: Inputs for winning/preferred responses
        lose_inputs: Inputs for losing/rejected responses
        beta: Temperature parameter controlling strength of KL penalty
    
    Returns:
        Tuple of (loss, outputs)
    
    Reference:
        Rafailov et al. "Direct Preference Optimization" NeurIPS 2023
    """
    # Compute log probabilities for winning responses
    win_logprobs = _get_batch_logps(model, win_inputs)
    with torch.no_grad():
        win_ref_logprobs = _get_batch_logps(ref_model, win_inputs)
    
    # Compute log probabilities for losing responses
    lose_logprobs = _get_batch_logps(model, lose_inputs)
    with torch.no_grad():
        lose_ref_logprobs = _get_batch_logps(ref_model, lose_inputs)
    
    # Compute log ratios
    win_logratios = win_logprobs - win_ref_logprobs
    lose_logratios = lose_logprobs - lose_ref_logprobs
    
    # DPO loss: -log(sigmoid(beta * (win_ratio - lose_ratio)))
    logits = beta * (win_logratios - lose_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    # Get outputs for return
    outputs = model(**win_inputs)
    
    return loss, outputs


def _get_batch_logps(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute log probabilities for a batch of sequences.
    
    Args:
        model: The model
        inputs: Input dictionary with input_ids, attention_mask, labels
    
    Returns:
        Log probability for each sequence in the batch
    """
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for the actual labels
    per_token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask out padding tokens
    loss_mask = (shift_labels != -100).float()
    per_token_logps = per_token_logps * loss_mask
    
    # Sum log probs for each sequence
    sequence_logps = per_token_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
    
    return sequence_logps


def compute_nll_loss(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute standard negative log-likelihood loss.
    
    Args:
        model: The model
        inputs: Input dictionary
        reduction: How to reduce the loss ('mean', 'sum', 'none')
    
    Returns:
        NLL loss
    """
    outputs = model(**inputs)
    return outputs.loss


def compute_entropy_loss(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute entropy of model predictions.
    
    Higher entropy = more uncertain predictions
    Useful for untargeted unlearning.
    
    Args:
        model: The model
        inputs: Input dictionary
    
    Returns:
        Negative entropy (to maximize entropy via minimization)
    """
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()
    
    # Compute probabilities
    probs = F.softmax(shift_logits, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)
    
    # Mask out padding
    loss_mask = (shift_labels != -100).float()
    entropy = entropy * loss_mask
    
    # Average over non-masked tokens
    avg_entropy = entropy.sum() / loss_mask.sum().clamp(min=1)
    
    # Return negative entropy (to maximize via minimization)
    return -avg_entropy