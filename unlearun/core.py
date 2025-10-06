# unlearun/core.py
"""Core Unlearning class for high-level API."""

import os
import logging
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset as HFDataset
from accelerate import Accelerator

from .data.dataset import UnlearningDataset
from .data.collators import DataCollatorForUnlearning
from .methods import get_unlearning_method
from .trainer.trainer import UnlearningTrainer
from .evaluation.metrics import evaluate_unlearning

logger = logging.getLogger(__name__)

class Unlearning:
    """
    High-level interface for machine unlearning in LLMs.
    
    Args:
        method (str): Unlearning method to use. Options: 'grad_ascent', 'grad_diff', 'dpo', 'rmu', 'simnpo'
        model (str or PreTrainedModel): Model name or path, or a pre-loaded model
        tokenizer (str or PreTrainedTokenizer, optional): Tokenizer name or path, or a pre-loaded tokenizer
        device (str): Device to use ('cuda', 'cpu', 'auto')
        seed (int): Random seed for reproducibility
        output_dir (str): Directory to save outputs
        **kwargs: Additional arguments passed to the unlearning method
    """
    
    SUPPORTED_METHODS = {
        'grad_ascent': 'GradAscent',
        'grad_diff': 'GradDiff',
        'dpo': 'DPO',
        'rmu': 'RMU',
        'simnpo': 'SimNPO',
    }
    
    def __init__(
        self,
        method: str,
        model: Union[str, Any],
        tokenizer: Optional[Union[str, Any]] = None,
        device: str = "auto",
        seed: int = 42,
        output_dir: str = "./unlearning_outputs",
        **kwargs
    ):
        self.method = method.lower()
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: {list(self.SUPPORTED_METHODS.keys())}"
            )
        
        # Set seed for reproducibility
        set_seed(seed)
        self.seed = seed
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(model, tokenizer, device)
        
        # Set output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store method-specific kwargs
        self.method_kwargs = kwargs
        
        # Initialize datasets
        self.forget_dataset = None
        self.retain_dataset = None
        self.holdout_dataset = None
        
        # Initialize trainer
        self.trainer = None
        
        logger.info(f"Initialized Unlearning with method: {self.method}")
    
    def _load_model_and_tokenizer(
        self, 
        model: Union[str, Any], 
        tokenizer: Optional[Union[str, Any]],
        device: str
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        # Load model
        if isinstance(model, str):
            logger.info(f"Loading model from: {model}")
            loaded_model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None,
            )
        else:
            loaded_model = model
        
        # Load tokenizer
        if tokenizer is None:
            if isinstance(model, str):
                tokenizer_name = model
            else:
                raise ValueError("Tokenizer must be provided if model is pre-loaded")
        else:
            tokenizer_name = tokenizer
        
        if isinstance(tokenizer_name, str):
            logger.info(f"Loading tokenizer from: {tokenizer_name}")
            loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            loaded_tokenizer = tokenizer_name
        
        # Set padding token if not set
        if loaded_tokenizer.pad_token is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
        
        return loaded_model, loaded_tokenizer
    
    def load_data(
        self,
        forget_data: Union[str, HFDataset, List[Dict[str, str]]],
        retain_data: Union[str, HFDataset, List[Dict[str, str]]],
        holdout_data: Optional[Union[str, HFDataset, List[Dict[str, str]]]] = None,
        question_key: str = "question",
        answer_key: str = "answer",
        max_length: int = 512,
        **kwargs
    ):
        """
        Load datasets for unlearning.
        
        Args:
            forget_data: Data to forget (path, HuggingFace dataset, or list of dicts)
            retain_data: Data to retain (path, HuggingFace dataset, or list of dicts)
            holdout_data: Optional holdout data for evaluation
            question_key: Key for questions in the data
            answer_key: Key for answers in the data
            max_length: Maximum sequence length
            **kwargs: Additional arguments for dataset creation
        """
        logger.info("Loading datasets...")
        
        # Create forget dataset
        self.forget_dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=self.tokenizer,
            question_key=question_key,
            answer_key=answer_key,
            max_length=max_length,
            dataset_type="forget",
            **kwargs
        )
        
        # Create retain dataset
        self.retain_dataset = UnlearningDataset(
            data=retain_data,
            tokenizer=self.tokenizer,
            question_key=question_key,
            answer_key=answer_key,
            max_length=max_length,
            dataset_type="retain",
            **kwargs
        )
        
        # Create holdout dataset if provided
        if holdout_data is not None:
            self.holdout_dataset = UnlearningDataset(
                data=holdout_data,
                tokenizer=self.tokenizer,
                question_key=question_key,
                answer_key=answer_key,
                max_length=max_length,
                dataset_type="holdout",
                **kwargs
            )
        
        logger.info(f"Loaded datasets - Forget: {len(self.forget_dataset)}, "
                   f"Retain: {len(self.retain_dataset)}, "
                   f"Holdout: {len(self.holdout_dataset) if self.holdout_dataset else 0}")
    
    def configure_training(
        self,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        **kwargs
    ) -> TrainingArguments:
        """
        Configure training arguments.
        
        Returns:
            TrainingArguments: Configured training arguments
        """
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps" if self.holdout_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if self.holdout_dataset else False,
            metric_for_best_model="eval_loss" if self.holdout_dataset else None,
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"],
            seed=self.seed,
            **kwargs
        )
        
        return training_args
    
    def run(
        self,
        training_args: Optional[TrainingArguments] = None,
        **kwargs
    ):
        """
        Run the unlearning process.
        
        Args:
            training_args: Training arguments (if None, uses default configuration)
            **kwargs: Additional arguments passed to configure_training
        """
        if self.forget_dataset is None or self.retain_dataset is None:
            raise ValueError("Datasets must be loaded before running unlearning. "
                           "Call load_data() first.")
        
        # Configure training if not provided
        if training_args is None:
            training_args = self.configure_training(**kwargs)
        
        # Get unlearning method class
        method_class = get_unlearning_method(self.SUPPORTED_METHODS[self.method])
        
        # Create data collator
        data_collator = DataCollatorForUnlearning(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.forget_dataset.max_length,
        )
        
        # Initialize trainer with the specific unlearning method
        self.trainer = UnlearningTrainer(
            method_class=method_class,
            method_kwargs=self.method_kwargs,
            model=self.model,
            args=training_args,
            train_dataset=self.forget_dataset,
            retain_dataset=self.retain_dataset,
            eval_dataset=self.holdout_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        logger.info(f"Starting unlearning with {self.method}...")
        
        # Train
        train_result = self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        
        # Save training results
        with open(self.output_dir / "train_results.txt", "w") as f:
            f.write(str(train_result))
        
        logger.info(f"Unlearning completed! Model saved to {self.output_dir}")
        
        return train_result
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[str, HFDataset, List[Dict[str, str]]]] = None,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the unlearned model.
        
        Args:
            eval_dataset: Dataset to evaluate on (if None, uses holdout dataset)
            metrics: List of metrics to compute
            **kwargs: Additional arguments for evaluation
        
        Returns:
            Dict[str, float]: Evaluation results
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation. Call run() first.")
        
        # Use holdout dataset if no eval dataset provided
        if eval_dataset is None:
            if self.holdout_dataset is None:
                raise ValueError("No evaluation dataset provided and no holdout dataset loaded.")
            eval_dataset = self.holdout_dataset
        elif not isinstance(eval_dataset, UnlearningDataset):
            # Convert to UnlearningDataset if needed
            eval_dataset = UnlearningDataset(
                data=eval_dataset,
                tokenizer=self.tokenizer,
                max_length=self.forget_dataset.max_length,
                dataset_type="eval",
                **kwargs
            )
        
        # Default metrics if none provided
        if metrics is None:
            metrics = ["perplexity", "rouge", "forget_quality"]
        
        logger.info(f"Evaluating model on {len(eval_dataset)} examples...")
        
        # Run evaluation
        results = evaluate_unlearning(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataset=eval_dataset,
            forget_dataset=self.forget_dataset,
            retain_dataset=self.retain_dataset,
            metrics=metrics,
            device=self.accelerator.device,
            **kwargs
        )
        
        # Save results
        import json
        with open(self.output_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed! Results saved to {self.output_dir}")
        
        return results
    
    def save_model(self, path: Optional[str] = None):
        """Save the unlearned model."""
        if path is None:
            path = self.output_dir / "final_model"
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a previously unlearned model."""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        logger.info(f"Model loaded from {path}")