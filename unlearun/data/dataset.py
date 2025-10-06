# unlearun/data/dataset.py
"""Dataset classes for unlearning."""

import json
import logging
from pathlib import Path
from typing import Union, Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class UnlearningDataset(Dataset):
    """
    Dataset for unlearning that handles question-answer pairs.
    
    Args:
        data: Data source (path, HuggingFace dataset, or list of dicts)
        tokenizer: Tokenizer to use
        question_key: Key for questions in the data
        answer_key: Key for answers in the data
        max_length: Maximum sequence length
        dataset_type: Type of dataset ('forget', 'retain', 'holdout', 'eval')
    """
    
    def __init__(
        self,
        data: Union[str, HFDataset, List[Dict[str, str]]],
        tokenizer: PreTrainedTokenizer,
        question_key: str = "question",
        answer_key: str = "answer",
        max_length: int = 512,
        dataset_type: str = "unknown",
        add_prefix: bool = True,
        instruction_template: Optional[str] = None,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.question_key = question_key
        self.answer_key = answer_key
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.add_prefix = add_prefix
        self.instruction_template = instruction_template or "Question: {question}\nAnswer:"
        
        # Load data
        self.data = self._load_data(data, **kwargs)
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Loaded {len(self.data)} examples for {dataset_type} dataset")
    
    def _load_data(self, data: Union[str, HFDataset, List[Dict[str, str]]], **kwargs) -> List[Dict[str, str]]:
        """Load data from various sources."""
        if isinstance(data, str):
            # Load from file or HuggingFace
            path = Path(data)
            if path.exists() and path.suffix == '.json':
                with open(path, 'r') as f:
                    loaded_data = json.load(f)
            elif path.exists() and path.suffix == '.jsonl':
                loaded_data = []
                with open(path, 'r') as f:
                    for line in f:
                        loaded_data.append(json.loads(line.strip()))
            else:
                # Try loading from HuggingFace
                dataset = load_dataset(data, **kwargs)
                if isinstance(dataset, dict):
                    # Use train split by default
                    dataset = dataset.get('train', list(dataset.values())[0])
                loaded_data = list(dataset)
        elif isinstance(data, HFDataset):
            loaded_data = list(data)
        elif isinstance(data, list):
            loaded_data = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return loaded_data
    
    def _validate_data(self):
        """Validate that data has required keys."""
        if not self.data:
            raise ValueError(f"No data loaded for {self.dataset_type} dataset")
        
        # Check first item
        first_item = self.data[0]
        if self.question_key not in first_item:
            raise ValueError(f"Question key '{self.question_key}' not found in data")
        if self.answer_key not in first_item:
            raise ValueError(f"Answer key '{self.answer_key}' not found in data")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        item = self.data[idx]
        question = item[self.question_key]
        answer = item[self.answer_key]
        
        # Format input
        if self.add_prefix:
            input_text = self.instruction_template.format(question=question)
            full_text = f"{input_text} {answer}"
        else:
            input_text = question
            full_text = f"{question} {answer}"
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Create labels (mask input portion)
        labels = full_encoding["input_ids"].copy()
        input_len = len(input_encoding["input_ids"])
        labels[:input_len] = [-100] * input_len  # Mask input tokens
        
        # Add EOS token if not present
        if labels[-1] != self.tokenizer.eos_token_id:
            if len(labels) < self.max_length:
                full_encoding["input_ids"].append(self.tokenizer.eos_token_id)
                full_encoding["attention_mask"].append(1)
                labels.append(self.tokenizer.eos_token_id)
            else:
                # Replace last token with EOS
                full_encoding["input_ids"][-1] = self.tokenizer.eos_token_id
                labels[-1] = self.tokenizer.eos_token_id
        
        return {
            "input_ids": torch.tensor(full_encoding["input_ids"]),
            "attention_mask": torch.tensor(full_encoding["attention_mask"]),
            "labels": torch.tensor(labels),
            "index": idx,
        }


class ForgetRetainDataset(Dataset):
    """
    Wrapper dataset that pairs forget and retain examples for unlearning.
    
    Args:
        forget_dataset: Dataset containing examples to forget
        retain_dataset: Dataset containing examples to retain
        anchor: Which dataset to anchor ('forget' or 'retain')
    """
    
    def __init__(
        self,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        anchor: str = "forget"
    ):
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.anchor = anchor
        
        if anchor not in ["forget", "retain"]:
            raise ValueError(f"anchor must be 'forget' or 'retain', got {anchor}")
        
        # Validate datasets
        if anchor == "forget" and not forget_dataset:
            raise ValueError("forget_dataset cannot be None when anchor='forget'")
        if anchor == "retain" and not retain_dataset:
            raise ValueError("retain_dataset cannot be None when anchor='retain'")
    
    def __len__(self) -> int:
        """Return length based on anchor dataset."""
        if self.anchor == "forget":
            return len(self.forget_dataset)
        else:
            return len(self.retain_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get paired examples from forget and retain datasets."""
        item = {}
        
        if self.anchor == "forget":
            item["forget"] = self.forget_dataset[idx]
            if self.retain_dataset:
                # Randomly sample from retain dataset
                retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
                item["retain"] = self.retain_dataset[retain_idx]
        else:
            item["retain"] = self.retain_dataset[idx]
            if self.forget_dataset:
                # Randomly sample from forget dataset
                forget_idx = torch.randint(0, len(self.forget_dataset), (1,)).item()
                item["forget"] = self.forget_dataset[forget_idx]
        
        return item