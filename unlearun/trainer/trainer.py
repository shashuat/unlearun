# unlearun/trainer/trainer.py
"""Trainer for unlearning methods."""

import logging
from typing import Optional, Dict, Any, List, Union, Callable

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..data import ForgetRetainDataset
from ..methods import BaseUnlearningMethod

logger = logging.getLogger(__name__)


class UnlearningTrainer(Trainer):
    """
    Custom trainer for unlearning methods.
    
    Extends HuggingFace Trainer to support unlearning-specific loss computation
    and data handling.
    """
    
    def __init__(
        self,
        method_class: type,
        method_kwargs: Dict[str, Any],
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        retain_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable] = None,
    ):
        # Create the forget/retain dataset
        if train_dataset is not None and retain_dataset is not None:
            train_dataset = ForgetRetainDataset(
                forget_dataset=train_dataset,
                retain_dataset=retain_dataset,
                anchor="forget"
            )
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # Initialize unlearning method
        self.unlearning_method = method_class(
            model=self.model,
            **method_kwargs
        )
        
        # Call method's initialization hook
        self.unlearning_method.on_train_begin()
    
    def compute_loss(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss using the unlearning method.
        
        Args:
            model: The model being trained
            inputs: Dictionary containing batch data
            return_outputs: Whether to return outputs
        
        Returns:
            Loss tensor or (loss, outputs) tuple
        """
        # During evaluation, use standard loss
        if not self.model.training:
            if "forget" in inputs:
                # Use only forget data for evaluation
                eval_inputs = inputs["forget"]
            else:
                eval_inputs = inputs
            
            outputs = model(**eval_inputs)
            if return_outputs:
                return outputs.loss, outputs
            return outputs.loss
        
        # During training, use unlearning method
        return self.unlearning_method.compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs
        )
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple:
        """
        Perform evaluation step.
        
        Override to handle forget/retain data structure during evaluation.
        """
        # Extract eval inputs
        if "forget" in inputs:
            eval_inputs = inputs["forget"]
        else:
            eval_inputs = inputs
        
        # Use parent's prediction step with cleaned inputs
        return super().prediction_step(
            model=model,
            inputs=eval_inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Get training dataloader.
        
        Override to ensure proper handling of ForgetRetainDataset.
        """
        train_dataloader = super().get_train_dataloader()
        
        # Log dataset info
        if isinstance(self.train_dataset, ForgetRetainDataset):
            logger.info(
                f"Training with {len(self.train_dataset.forget_dataset)} forget examples "
                f"and {len(self.train_dataset.retain_dataset)} retain examples"
            )
        
        return train_dataloader
    
    def create_optimizer(self):
        """
        Create optimizer with method-specific parameters.
        """
        # Get parameters from unlearning method
        opt_params = self.unlearning_method.get_optimizer_params(self.model)
        
        # Filter to only parameters that require gradients
        opt_params = [p for p in opt_params if p.requires_grad]
        
        if not opt_params:
            raise ValueError("No parameters require gradients!")
        
        # Log parameter info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in opt_params)
        logger.info(
            f"Total parameters: {total_params:,} | "
            f"Trainable parameters: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        
        # Create optimizer
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(opt_params, **optimizer_kwargs)
        
        return self.optimizer
    
    def training_step(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform a training step.
        
        Adds hooks for the unlearning method.
        
        Args:
            model: The model being trained
            inputs: Input batch
            num_items_in_batch: Number of items in the batch (for newer transformers versions)
        """
        # Call method's step begin hook
        if hasattr(self.state, 'global_step'):
            self.unlearning_method.on_step_begin(self.state.global_step)
        
        # Perform training step - pass num_items_in_batch if parent expects it
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            # Fallback for older transformers versions that don't expect num_items_in_batch
            loss = super().training_step(model, inputs)
        
        # Call method's step end hook
        if hasattr(self.state, 'global_step'):
            self.unlearning_method.on_step_end(self.state.global_step, loss.item())
        
        return loss
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save model and method-specific information."""
        super()._save(output_dir, state_dict)
        
        # Save method configuration
        if output_dir is not None:
            import json
            import os
            
            method_config = {
                "method_class": self.unlearning_method.__class__.__name__,
                "method_kwargs": self.unlearning_method.config,
            }
            
            config_path = os.path.join(output_dir, "unlearning_config.json")
            with open(config_path, "w") as f:
                json.dump(method_config, f, indent=2)
    
    def on_train_end(self):
        """Called at the end of training."""
        super().on_train_end()
        
        # Call method's train end hook
        self.unlearning_method.on_train_end()