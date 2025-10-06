# unlearun/data/collators.py
"""Data collators for unlearning."""

from typing import Dict, List, Any, Optional

import torch
from transformers import PreTrainedTokenizer


class DataCollatorForUnlearning:
    """
    Data collator for unlearning that handles forget/retain data pairs.
    
    Args:
        tokenizer: Tokenizer to use for padding
        padding: Whether to pad sequences
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of forget/retain pairs.
        
        Args:
            features: List of feature dictionaries
        
        Returns:
            Collated batch with 'forget' and 'retain' keys
        """
        # Check if features contain forget/retain structure
        if "forget" in features[0]:
            # Handle forget/retain pairs
            forget_features = [f["forget"] for f in features]
            retain_features = [f.get("retain") for f in features if "retain" in f]
            
            batch = {}
            batch["forget"] = self._collate_features(forget_features)
            
            if retain_features:
                batch["retain"] = self._collate_features(retain_features)
            
            return batch
        else:
            # Handle regular features
            return self._collate_features(features)
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a list of feature dictionaries."""
        # Extract input_ids, attention_mask, and labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences
        if self.padding:
            input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
            attention_mask = self._pad_sequence(attention_mask, 0)
            labels = self._pad_sequence(labels, -100)
        else:
            input_ids = torch.stack(input_ids)
            attention_mask = torch.stack(attention_mask)
            labels = torch.stack(labels)
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Add indices if present
        if "index" in features[0]:
            batch["index"] = torch.tensor([f["index"] for f in features])
        
        return batch
    
    def _pad_sequence(self, sequences: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        """Pad sequences to the same length."""
        # Find max length
        if self.max_length:
            max_len = self.max_length
        else:
            max_len = max(len(seq) for seq in sequences)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
        
        # Pad sequences
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = torch.full(
                    (max_len - len(seq),), 
                    padding_value, 
                    dtype=seq.dtype
                )
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq[:max_len]
            padded.append(padded_seq)
        
        return torch.stack(padded)


class DataCollatorForLanguageModeling:
    """
    Standard data collator for language modeling tasks.
    
    Args:
        tokenizer: Tokenizer to use for padding
        mlm: Whether to use masked language modeling
        mlm_probability: Probability of masking tokens (if mlm=True)
        pad_to_multiple_of: Pad to multiple of this value
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        
        if mlm:
            raise NotImplementedError("MLM is not yet implemented")
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        # Use the unlearning collator's logic
        collator = DataCollatorForUnlearning(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return collator(features)