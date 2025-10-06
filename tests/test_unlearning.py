# tests/test_unlearning.py
"""Comprehensive test suite for unlearun package."""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearun import Unlearning
from unlearun.data.dataset import UnlearningDataset, ForgetRetainDataset
from unlearun.data.collators import DataCollatorForUnlearning
from unlearun.methods import (
    GradAscent,
    GradDiff,
    DPO,
    RMU,
    SimNPO,
    get_unlearning_method
)
from unlearun.utils.losses import (
    compute_kl_divergence,
    compute_dpo_loss,
    compute_nll_loss,
    compute_entropy_loss
)
from unlearun.evaluation.metrics import (
    compute_perplexity,
    compute_forget_quality,
    compute_model_utility,
    compute_rouge_score,
    compute_verbatim_memorization,
    compute_mia
)


# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample forget/retain data."""
    forget_data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
        {"question": "What is 2+2?", "answer": "4"},
    ]
    
    retain_data = [
        {"question": "What is the capital of Germany?", "answer": "Berlin"},
        {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
        {"question": "What is 3+3?", "answer": "6"},
    ]
    
    return forget_data, retain_data


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tiny_model():
    """Load a tiny model for testing."""
    model_name = "gpt2"  # Small model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ============================================================================
# Dataset Tests
# ============================================================================

class TestUnlearningDataset:
    """Test UnlearningDataset class."""
    
    def test_dataset_initialization(self, sample_data, tiny_model):
        """Test dataset can be initialized."""
        forget_data, _ = sample_data
        _, tokenizer = tiny_model
        
        dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        assert len(dataset) == len(forget_data)
    
    def test_dataset_getitem(self, sample_data, tiny_model):
        """Test dataset __getitem__ returns correct format."""
        forget_data, _ = sample_data
        _, tokenizer = tiny_model
        
        dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
    
    def test_dataset_from_json(self, sample_data, tiny_model, temp_dir):
        """Test loading dataset from JSON file."""
        forget_data, _ = sample_data
        _, tokenizer = tiny_model
        
        # Save data to JSON
        json_path = temp_dir / "forget_data.json"
        with open(json_path, 'w') as f:
            json.dump(forget_data, f)
        
        # Load dataset
        dataset = UnlearningDataset(
            data=str(json_path),
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        assert len(dataset) == len(forget_data)
    
    def test_forget_retain_dataset(self, sample_data, tiny_model):
        """Test ForgetRetainDataset wrapper."""
        forget_data, retain_data = sample_data
        _, tokenizer = tiny_model
        
        forget_dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        retain_dataset = UnlearningDataset(
            data=retain_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="retain"
        )
        
        paired_dataset = ForgetRetainDataset(
            forget_dataset=forget_dataset,
            retain_dataset=retain_dataset,
            anchor="forget"
        )
        
        assert len(paired_dataset) == len(forget_dataset)
        
        item = paired_dataset[0]
        assert "forget" in item
        assert "retain" in item


# ============================================================================
# Data Collator Tests
# ============================================================================

class TestDataCollator:
    """Test data collators."""
    
    def test_collator_basic(self, sample_data, tiny_model):
        """Test basic collation."""
        forget_data, _ = sample_data
        _, tokenizer = tiny_model
        
        dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        collator = DataCollatorForUnlearning(tokenizer=tokenizer)
        
        # Get batch
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        collated = collator(batch)
        
        assert "input_ids" in collated
        assert "attention_mask" in collated
        assert "labels" in collated
        assert collated["input_ids"].shape[0] == len(batch)
    
    def test_collator_forget_retain(self, sample_data, tiny_model):
        """Test collation of forget/retain pairs."""
        forget_data, retain_data = sample_data
        _, tokenizer = tiny_model
        
        forget_dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        retain_dataset = UnlearningDataset(
            data=retain_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="retain"
        )
        
        paired_dataset = ForgetRetainDataset(
            forget_dataset=forget_dataset,
            retain_dataset=retain_dataset,
            anchor="forget"
        )
        
        collator = DataCollatorForUnlearning(tokenizer=tokenizer)
        
        batch = [paired_dataset[i] for i in range(min(2, len(paired_dataset)))]
        collated = collator(batch)
        
        assert "forget" in collated
        assert "retain" in collated


# ============================================================================
# Method Tests
# ============================================================================

class TestUnlearningMethods:
    """Test unlearning methods."""
    
    def test_grad_ascent(self, sample_data, tiny_model):
        """Test Gradient Ascent method."""
        forget_data, retain_data = sample_data
        model, tokenizer = tiny_model
        
        method = GradAscent(model=model)
        
        # Create mock inputs
        forget_dataset = UnlearningDataset(
            data=forget_data[:1],
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        inputs = {"forget": forget_dataset[0]}
        inputs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs["forget"].items()}
        
        # Compute loss
        loss = method.compute_loss(model, {"forget": inputs})
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_grad_diff(self, sample_data, tiny_model):
        """Test Gradient Difference method."""
        forget_data, retain_data = sample_data
        model, tokenizer = tiny_model
        
        method = GradDiff(model=model, gamma=1.0, alpha=1.0)
        
        # Create datasets
        forget_dataset = UnlearningDataset(
            data=forget_data[:1],
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        retain_dataset = UnlearningDataset(
            data=retain_data[:1],
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="retain"
        )
        
        forget_inputs = forget_dataset[0]
        forget_inputs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                        for k, v in forget_inputs.items()}
        
        retain_inputs = retain_dataset[0]
        retain_inputs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                        for k, v in retain_inputs.items()}
        
        # Compute loss
        loss = method.compute_loss(
            model, 
            {"forget": forget_inputs, "retain": retain_inputs}
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_simnpo(self, sample_data, tiny_model):
        """Test SimNPO method."""
        forget_data, retain_data = sample_data
        model, tokenizer = tiny_model
        
        method = SimNPO(model=model, delta=0.0, beta=1.0, gamma=1.0, alpha=1.0)
        
        # Create datasets
        forget_dataset = UnlearningDataset(
            data=forget_data[:1],
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        retain_dataset = UnlearningDataset(
            data=retain_data[:1],
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="retain"
        )
        
        forget_inputs = forget_dataset[0]
        forget_inputs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                        for k, v in forget_inputs.items()}
        
        retain_inputs = retain_dataset[0]
        retain_inputs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                        for k, v in retain_inputs.items()}
        
        # Compute loss
        loss = method.compute_loss(
            model, 
            {"forget": forget_inputs, "retain": retain_inputs}
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_method_registry(self):
        """Test method registry lookup."""
        method_class = get_unlearning_method("GradAscent")
        assert method_class == GradAscent
        
        method_class = get_unlearning_method("GradDiff")
        assert method_class == GradDiff
        
        with pytest.raises(ValueError):
            get_unlearning_method("NonExistentMethod")


# ============================================================================
# Loss Function Tests
# ============================================================================

class TestLossFunctions:
    """Test loss functions."""
    
    def test_kl_divergence(self, tiny_model):
        """Test KL divergence computation."""
        model, tokenizer = tiny_model
        ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Create sample inputs
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(model, ref_model, inputs)
        
        assert isinstance(kl_div, torch.Tensor)
        assert kl_div.item() >= 0  # KL divergence is non-negative
    
    def test_nll_loss(self, tiny_model):
        """Test NLL loss computation."""
        model, tokenizer = tiny_model
        
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        
        loss = compute_nll_loss(model, inputs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_entropy_loss(self, tiny_model):
        """Test entropy loss computation."""
        model, tokenizer = tiny_model
        
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        
        entropy = compute_entropy_loss(model, inputs)
        
        assert isinstance(entropy, torch.Tensor)


# ============================================================================
# Evaluation Metrics Tests
# ============================================================================

class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_perplexity(self, sample_data, tiny_model):
        """Test perplexity computation."""
        forget_data, _ = sample_data
        model, tokenizer = tiny_model
        
        dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        ppl = compute_perplexity(model, dataset, tokenizer, device="cpu", batch_size=1)
        
        assert isinstance(ppl, float)
        assert ppl > 0
    
    def test_forget_quality(self, sample_data, tiny_model):
        """Test forget quality metric."""
        forget_data, _ = sample_data
        model, tokenizer = tiny_model
        
        dataset = UnlearningDataset(
            data=forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        quality = compute_forget_quality(model, dataset, tokenizer, device="cpu", batch_size=1)
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
    
    def test_model_utility(self, sample_data, tiny_model):
        """Test model utility metric."""
        _, retain_data = sample_data
        model, tokenizer = tiny_model
        
        dataset = UnlearningDataset(
            data=retain_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="retain"
        )
        
        utility = compute_model_utility(model, dataset, tokenizer, device="cpu", batch_size=1)
        
        assert isinstance(utility, float)
        assert 0 <= utility <= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test full unlearning pipeline."""
    
    def test_full_pipeline_grad_ascent(self, sample_data, tiny_model, temp_dir):
        """Test complete unlearning pipeline with Gradient Ascent."""
        forget_data, retain_data = sample_data
        model, tokenizer = tiny_model
        
        # Initialize Unlearning
        unlearner = Unlearning(
            method="grad_ascent",
            model=model,
            tokenizer=tokenizer,
            output_dir=str(temp_dir),
            seed=42
        )
        
        # Load data
        unlearner.load_data(
            forget_data=forget_data,
            retain_data=retain_data,
            max_length=128
        )
        
        # Run unlearning (with minimal epochs for testing)
        training_args = unlearner.configure_training(
            batch_size=1,
            learning_rate=5e-5,
            num_epochs=1,  # Just 1 epoch for testing
            logging_steps=1,
            save_steps=100
        )
        
        result = unlearner.run(training_args)
        
        assert result is not None
        assert (temp_dir / "pytorch_model.bin").exists() or \
               any((temp_dir).glob("*.safetensors"))
    
    def test_save_and_load_model(self, sample_data, tiny_model, temp_dir):
        """Test saving and loading unlearned model."""
        forget_data, retain_data = sample_data
        model, tokenizer = tiny_model
        
        # Initialize and run unlearning
        unlearner = Unlearning(
            method="grad_ascent",
            model=model,
            tokenizer=tokenizer,
            output_dir=str(temp_dir / "model1"),
            seed=42
        )
        
        unlearner.load_data(
            forget_data=forget_data[:1],  # Minimal data
            retain_data=retain_data[:1],
            max_length=128
        )
        
        training_args = unlearner.configure_training(
            batch_size=1,
            num_epochs=1,
            logging_steps=1
        )
        
        unlearner.run(training_args)
        
        # Save model
        save_path = temp_dir / "saved_model"
        unlearner.save_model(str(save_path))
        
        assert (save_path).exists()
        
        # Load model
        new_unlearner = Unlearning(
            method="grad_ascent",
            model=str(save_path),
            tokenizer=str(save_path),
            output_dir=str(temp_dir / "model2")
        )
        
        assert new_unlearner.model is not None
        assert new_unlearner.tokenizer is not None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError):
            Unlearning(
                method="invalid_method",
                model="gpt2",
                tokenizer="gpt2"
            )
    
    def test_run_without_data(self, tiny_model, temp_dir):
        """Test running unlearning without loading data."""
        model, tokenizer = tiny_model
        
        unlearner = Unlearning(
            method="grad_ascent",
            model=model,
            tokenizer=tokenizer,
            output_dir=str(temp_dir)
        )
        
        with pytest.raises(ValueError):
            unlearner.run()
    
    def test_evaluate_without_training(self, sample_data, tiny_model, temp_dir):
        """Test evaluation without training."""
        forget_data, retain_data = sample_data
        model, tokenizer = tiny_model
        
        unlearner = Unlearning(
            method="grad_ascent",
            model=model,
            tokenizer=tokenizer,
            output_dir=str(temp_dir)
        )
        
        unlearner.load_data(
            forget_data=forget_data,
            retain_data=retain_data
        )
        
        with pytest.raises(ValueError):
            unlearner.evaluate()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_batch_processing(self, sample_data, tiny_model):
        """Test that batch processing works correctly."""
        forget_data, _ = sample_data
        model, tokenizer = tiny_model
        
        dataset = UnlearningDataset(
            data=forget_data * 10,  # Duplicate to get more samples
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        collator = DataCollatorForUnlearning(tokenizer=tokenizer)
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]
            collated = collator(batch)
            
            assert collated["input_ids"].shape[0] == len(batch)
    
    @pytest.mark.slow
    def test_memory_efficiency(self, sample_data, tiny_model):
        """Test memory efficiency with larger datasets."""
        # This test is marked as slow and may be skipped
        forget_data, retain_data = sample_data
        model, tokenizer = tiny_model
        
        # Create larger dataset
        large_forget_data = forget_data * 100
        large_retain_data = retain_data * 100
        
        dataset = UnlearningDataset(
            data=large_forget_data,
            tokenizer=tokenizer,
            max_length=128,
            dataset_type="forget"
        )
        
        assert len(dataset) == len(large_forget_data)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])