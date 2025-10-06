# unlearun/methods/rmu.py
"""Representation Misdirection for Unlearning (RMU) method."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .base import BaseUnlearningMethod


class RMU(BaseUnlearningMethod):
    """
    Representation Misdirection for Unlearning.
    
    RMU steers the internal representations of forget samples toward random
    directions while keeping retain sample representations unchanged.
    
    Args:
        model: The model to be unlearned
        ref_model: Reference model (frozen copy of original)
        steering_coeff: Coefficient for steering strength (c in the paper)
        target_layer: Which layer to apply steering to (default: 8)
        random_vector: Optional pre-computed random vector
        adaptive: Whether to use adaptive RMU variant
        alpha: Weight for retain loss
    
    References:
        Li et al. "The WMDP Benchmark: Measuring and Reducing Malicious Use 
        with Unlearning" ICML 2024
        Dang et al. "On Effects of Steering Latent Representation for Large 
        Language Model Unlearning" AAAI 2025
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        steering_coeff: float = 1.0,
        target_layer: int = 8,
        random_vector: Optional[torch.Tensor] = None,
        adaptive: bool = False,
        alpha: float = 1.0,
        **kwargs
    ):
        super().__init__(model, ref_model, **kwargs)
        self.steering_coeff = steering_coeff
        self.target_layer = target_layer
        self.random_vector = random_vector
        self.adaptive = adaptive
        self.alpha = alpha
        
        # Store activations during forward pass
        self.forget_activations = None
        self.retain_activations = None
        
        # Register hooks for capturing activations
        self._register_hooks()
    
    @property
    def requires_ref_model(self) -> bool:
        """RMU requires a reference model."""
        return True
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        self.hooks = []
        
        def get_activation_hook(name):
            def hook(module, input, output):
                # Store the activation
                if hasattr(self, f'{name}_activations'):
                    setattr(self, f'{name}_activations', output[0])
            return hook
        
        # Register hook on the target layer
        # This assumes a transformer architecture with layers accessible as model.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            target_module = self.model.model.layers[self.target_layer]
            self.hooks.append(
                target_module.register_forward_hook(get_activation_hook('current'))
            )
    
    def _get_random_vector(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Generate or retrieve random vector for steering.
        
        Args:
            activation: The activation tensor to match dimensions
        
        Returns:
            Random unit vector of same shape as activation
        """
        if self.random_vector is None:
            # Generate random vector with same shape as activation
            random_vec = torch.randn_like(activation)
            # Normalize to unit vector
            random_vec = random_vec / (random_vec.norm(dim=-1, keepdim=True) + 1e-8)
            return random_vec
        else:
            return self.random_vector.to(activation.device)
    
    def _compute_adaptive_coeff(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive steering coefficient based on activation norm.
        
        Adaptive RMU adjusts coefficient based on the norm of forget representation.
        """
        if not self.adaptive:
            return self.steering_coeff
        
        # Compute norm of activation
        activation_norm = activation.norm(dim=-1, keepdim=True)
        
        # Scale coefficient inversely with norm
        adaptive_coeff = self.steering_coeff * activation_norm
        
        return adaptive_coeff
    
    def _compute_steering_loss(
        self,
        activation: torch.Tensor,
        random_vector: torch.Tensor,
        coeff: float
    ) -> torch.Tensor:
        """
        Compute the steering loss (MSE to random vector).
        
        Args:
            activation: Current activation
            random_vector: Target random direction
            coeff: Steering coefficient
        
        Returns:
            Steering loss
        """
        target = activation.detach() + coeff * random_vector
        loss = nn.functional.mse_loss(activation, target)
        return loss
    
    def _compute_retain_regularization(
        self,
        model_activation: torch.Tensor,
        ref_activation: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regularization loss to keep retain activations unchanged.
        
        Args:
            model_activation: Activation from current model
            ref_activation: Activation from reference model
        
        Returns:
            MSE loss between activations
        """
        return nn.functional.mse_loss(model_activation, ref_activation.detach())
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Any:
        """
        Compute RMU loss.
        
        The loss consists of:
        1. Steering loss: Pushes forget activations toward random direction
        2. Retain loss: Keeps retain activations close to reference model
        """
        total_loss = 0.0
        
        # Process forget data
        forget_inputs = self.prepare_inputs(inputs["forget"])
        
        # Forward pass on forget data
        self.forget_activations = None
        forget_outputs = model(**forget_inputs)
        
        if self.forget_activations is not None:
            # Get random vector
            random_vec = self._get_random_vector(self.forget_activations)
            
            # Get steering coefficient (adaptive or fixed)
            coeff = self._compute_adaptive_coeff(self.forget_activations)
            
            # Compute steering loss
            steering_loss = self._compute_steering_loss(
                self.forget_activations,
                random_vec,
                coeff
            )
            
            total_loss += steering_loss
        
        # Process retain data if available
        if "retain" in inputs:
            retain_inputs = self.prepare_inputs(inputs["retain"])
            
            # Get reference model activations
            with torch.no_grad():
                self.retain_activations = None
                ref_retain_outputs = self.ref_model(**retain_inputs)
                ref_retain_activation = self.retain_activations
            
            # Get current model activations
            self.retain_activations = None
            retain_outputs = model(**retain_inputs)
            curr_retain_activation = self.retain_activations
            
            if curr_retain_activation is not None and ref_retain_activation is not None:
                # Compute regularization loss
                retain_loss = self._compute_retain_regularization(
                    curr_retain_activation,
                    ref_retain_activation
                )
                
                total_loss += self.alpha * retain_loss
        
        return (total_loss, forget_outputs) if return_outputs else total_loss
    
    def get_optimizer_params(self, model: nn.Module) -> list:
        """
        Get parameters to optimize.
        
        RMU typically only optimizes certain layers, particularly MLP output projections.
        """
        # Option 1: Optimize only target layer and subsequent layers
        params = []
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                if i >= self.target_layer:
                    params.extend(layer.parameters())
        
        # Option 2: Optimize all parameters (simpler, may work better)
        # Uncomment this if option 1 doesn't work well
        # params = list(model.parameters())
        
        return params if params else list(model.parameters())
    
    def __del__(self):
        """Clean up hooks when object is deleted."""
        for hook in getattr(self, 'hooks', []):
            hook.remove()