# methods/base.py
"""Base class for unlearning methods."""

import copy
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class BaseUnlearningMethod(ABC):
    """
    Abstract base class for unlearning methods.
    
    All unlearning methods should inherit from this class and implement
    the compute_loss method.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel] = None,
        **kwargs
    ):
        """
        Initialize the unlearning method.
        
        Args:
            model: The model to be unlearned
            ref_model: Optional reference model (if None, will be created as a copy)
            **kwargs: Additional method-specific arguments
        """
        self.model = model
        self.ref_model = ref_model
        self.config = kwargs
        
        # Initialize reference model if needed by the method
        if self.requires_ref_model and self.ref_model is None:
            self.ref_model = self._create_ref_model(model)
    
    @property
    def requires_ref_model(self) -> bool:
        """Whether this method requires a reference model."""
        return False
    
    def _create_ref_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Create a reference model as a frozen copy of the original."""
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
    
    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Any:
        """
        Compute the loss for unlearning.
        
        Args:
            model: The model being trained
            inputs: Dictionary containing 'forget' and 'retain' data
            return_outputs: Whether to return model outputs along with loss
        
        Returns:
            Loss value or (loss, outputs) tuple if return_outputs=True
        """
        raise NotImplementedError
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.
        
        Args:
            batch: Raw batch from dataloader
        
        Returns:
            Processed inputs ready for the model
        """
        # Default implementation - override if needed
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch.get("labels", batch["input_ids"]),
        }
    
    def get_optimizer_params(self, model: nn.Module) -> list:
        """
        Get parameters to optimize.
        
        Args:
            model: The model being trained
        
        Returns:
            List of parameters to optimize
        """
        # Default: all parameters - override for methods like RMU
        return list(model.parameters())
    
    def on_train_begin(self):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self):
        """Called at the end of training."""
        pass
    
    def on_step_begin(self, step: int):
        """Called at the beginning of each training step."""
        pass
    
    def on_step_end(self, step: int, loss: float):
        """Called at the end of each training step."""
        pass