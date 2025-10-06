# unlearun/evaluation/metrics.py
"""Evaluation metrics for unlearning."""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from rouge_score import rouge_scorer


def evaluate_unlearning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Any,
    forget_dataset: Optional[Any] = None,
    retain_dataset: Optional[Any] = None,
    metrics: List[str] = None,
    device: str = "cuda",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    **kwargs
) -> Dict[str, float]:
    """
    Comprehensive evaluation of unlearned model.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Dataset to evaluate on
        forget_dataset: Forget dataset for comparison
        retain_dataset: Retain dataset for comparison
        metrics: List of metrics to compute
        device: Device to use
        batch_size: Batch size for evaluation
        max_new_tokens: Max tokens to generate
    
    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = [
            "perplexity",
            "forget_quality",
            "model_utility",
            "rouge",
        ]
    
    results = {}
    model.eval()
    model.to(device)
    
    # Compute requested metrics
    if "perplexity" in metrics:
        results["perplexity"] = compute_perplexity(
            model, eval_dataset, tokenizer, device, batch_size
        )
    
    if "forget_quality" in metrics and forget_dataset is not None:
        results["forget_quality"] = compute_forget_quality(
            model, forget_dataset, tokenizer, device, batch_size
        )
    
    if "model_utility" in metrics and retain_dataset is not None:
        results["model_utility"] = compute_model_utility(
            model, retain_dataset, tokenizer, device, batch_size
        )
    
    if "rouge" in metrics:
        results["rouge"] = compute_rouge_score(
            model, eval_dataset, tokenizer, device, max_new_tokens
        )
    
    if "verbatim_memorization" in metrics:
        results["verbatim_memorization"] = compute_verbatim_memorization(
            model, forget_dataset, tokenizer, device, max_new_tokens
        )
    
    if "mia" in metrics and forget_dataset and retain_dataset:
        results["mia_score"] = compute_mia(
            model, forget_dataset, retain_dataset, tokenizer, device, batch_size
        )
    
    return results


def compute_perplexity(
    model: PreTrainedModel,
    dataset: Any,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    batch_size: int = 4
) -> float:
    """
    Compute perplexity on a dataset.
    
    Lower perplexity = better language modeling (for retain set)
    Higher perplexity = better forgetting (for forget set)
    """
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            
            # Accumulate loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                # Count non-padded tokens
                num_tokens = (batch["labels"] != -100).sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return perplexity


def compute_forget_quality(
    model: PreTrainedModel,
    forget_dataset: Any,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    batch_size: int = 4
) -> float:
    """
    Compute forget quality score.
    
    Higher score = better forgetting
    Based on perplexity increase on forget set.
    """
    # Compute perplexity on forget set
    forget_ppl = compute_perplexity(
        model, forget_dataset, tokenizer, device, batch_size
    )
    
    # Normalize to 0-1 scale (higher = better forgetting)
    # You can adjust the normalization based on expected ranges
    # For now, use inverse perplexity
    forget_quality = 1.0 / (1.0 + np.log(forget_ppl))
    
    return forget_quality


def compute_model_utility(
    model: PreTrainedModel,
    retain_dataset: Any,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    batch_size: int = 4
) -> float:
    """
    Compute model utility score.
    
    Higher score = better retention of useful knowledge
    Based on perplexity on retain set.
    """
    # Compute perplexity on retain set
    retain_ppl = compute_perplexity(
        model, retain_dataset, tokenizer, device, batch_size
    )
    
    # Convert to utility score (lower perplexity = higher utility)
    utility = 1.0 / (1.0 + np.log(retain_ppl))
    
    return utility


def compute_rouge_score(
    model: PreTrainedModel,
    dataset: Any,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    max_new_tokens: int = 128,
    num_samples: int = 50
) -> Dict[str, float]:
    """
    Compute ROUGE scores between generated and ground truth text.
    
    Used for evaluating verbatim memorization.
    """

    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    # Sample from dataset
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing ROUGE"):
            item = dataset[int(idx)]
            
            # Get input
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Get reference (ground truth continuation)
            labels = item["labels"]
            reference_ids = labels[labels != -100]
            reference = tokenizer.decode(reference_ids, skip_special_tokens=True)
            
            # Compute ROUGE
            rouge_scores = scorer.score(reference, generated)
            
            for key in scores:
                scores[key].append(rouge_scores[key].fmeasure)
    
    # Average scores
    return {key: np.mean(values) for key, values in scores.items()}


def compute_verbatim_memorization(
    model: PreTrainedModel,
    forget_dataset: Any,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    max_new_tokens: int = 128,
    num_samples: int = 50,
    prefix_length: int = 50
) -> float:
    """
    Compute verbatim memorization score.
    
    Measures how much the model reproduces exact text from forget set.
    Lower score = better unlearning.
    
    Method: Give model first N tokens, measure overlap with continuation.
    """

    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    overlap_scores = []
    
    # Sample from dataset
    indices = np.random.choice(len(forget_dataset), min(num_samples, len(forget_dataset)), replace=False)
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing verbatim memorization"):
            item = forget_dataset[int(idx)]
            
            # Get first N tokens as prefix
            input_ids = item["input_ids"]
            if len(input_ids) < prefix_length:
                continue
            
            prefix_ids = input_ids[:prefix_length].unsqueeze(0).to(device)
            
            # Generate continuation
            outputs = model.generate(
                input_ids=prefix_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Get generated continuation (excluding prefix)
            generated_ids = outputs[0][prefix_length:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Get true continuation
            true_continuation_ids = input_ids[prefix_length:prefix_length + max_new_tokens]
            true_continuation = tokenizer.decode(true_continuation_ids, skip_special_tokens=True)
            
            # Compute exact match ratio
            score = scorer.score(true_continuation, generated)
            overlap_scores.append(score['rougeL'].fmeasure)
    
    return np.mean(overlap_scores) if overlap_scores else 0.0


def compute_mia(
    model: PreTrainedModel,
    forget_dataset: Any,
    retain_dataset: Any,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    batch_size: int = 4
) -> float:
    """
    Compute Membership Inference Attack (MIA) score.
    
    Measures privacy leakage - can we detect if data was in training set?
    Lower score = better privacy (harder to detect membership).
    
    Method: Use loss-based MIA - members typically have lower loss.
    Returns AUROC where 0.5 = random guessing (best for privacy).
    """
    from torch.utils.data import DataLoader
    
    def get_losses(dataset):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch)
                
                # Get per-example loss
                if hasattr(outputs, 'loss'):
                    # This is average loss, we need per-example
                    logits = outputs.logits
                    labels = batch["labels"]
                    
                    # Compute per-example loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    per_token_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Reshape and average per example
                    per_token_loss = per_token_loss.view(labels.size(0), -1)
                    mask = (shift_labels != -100).float()
                    per_example_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1)
                    
                    losses.extend(per_example_loss.cpu().tolist())
        
        return losses
    
    # Get losses for forget (members) and retain (non-members for this test)
    forget_losses = get_losses(forget_dataset)
    retain_losses = get_losses(retain_dataset)
    
    # Create labels (1 = member/forget, 0 = non-member/retain)
    y_true = [1] * len(forget_losses) + [0] * len(retain_losses)
    
    # Use negative loss as score (lower loss = higher probability of membership)
    y_score = [-l for l in forget_losses] + [-l for l in retain_losses]
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = 0.5  # Random guessing
    
    return auroc


def compute_knowledge_retention_qa(
    model: PreTrainedModel,
    qa_dataset: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    max_new_tokens: int = 50
) -> float:
    """
    Compute QA accuracy to measure knowledge retention.
    
    Args:
        model: The model
        qa_dataset: List of dicts with 'question' and 'answer' keys
        tokenizer: Tokenizer
        device: Device
        max_new_tokens: Max tokens to generate
    
    Returns:
        Accuracy score (0-1)
    """
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for item in tqdm(qa_dataset, desc="Evaluating QA"):
            question = item["question"]
            true_answer = item["answer"].lower().strip()
            
            # Tokenize question
            inputs = tokenizer(question, return_tensors="pt").to(device)
            
            # Generate answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated[len(question):].lower().strip()
            
            # Check if answer is correct (simple substring match)
            if true_answer in generated_answer or generated_answer in true_answer:
                correct += 1
            
            total += 1
    
    return correct / total if total > 0 else 0.0