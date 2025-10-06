# Unlearun: Machine Unlearning for Fine-tuned LLMs

A comprehensive Python package for machine unlearning in large language models, enabling efficient removal of unwanted knowledge while preserving model utility.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Overview

**Unlearun** addresses the critical need to remove specific knowledge from trained language models without expensive retraining. This is essential for:

- **Privacy Compliance**: GDPR "right to be forgotten" requirements
- **Copyright Protection**: Removing copyrighted content from models
- **AI Safety**: Eliminating harmful or dangerous knowledge
- **Model Correction**: Fixing outdated or incorrect information

## ✨ Key Features

- **5 State-of-the-Art Methods**: GradAscent, GradDiff, DPO, RMU, SimNPO
- **Simple High-Level API**: Get started with just a few lines of code
- **Comprehensive Evaluation**: Built-in metrics for forget quality, utility preservation, and privacy
- **Flexible Data Loading**: Support for JSON, JSONL, HuggingFace datasets, and Python lists
- **Production Ready**: Extensive test coverage and benchmarking
- **HuggingFace Integration**: Seamless integration with `transformers` and `accelerate`

## 🚀 Quick Start

### Installation

```bash
pip install unlearun
```

Or install from source:

```bash
git clone https://github.com/shashuat/unlearun.git
cd unlearun
pip install -e .
```

### Basic Usage

```python
from unlearun import Unlearning

# Initialize unlearner with RMU method
unlearner = Unlearning(
    method="rmu",
    model="gpt2-medium",
    output_dir="./unlearned_model"
)

# Load your data
forget_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"}
]

retain_data = [
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"}
]

unlearner.load_data(
    forget_data=forget_data,
    retain_data=retain_data,
    max_length=128
)

# Run unlearning
unlearner.run(
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3
)

# Evaluate results
results = unlearner.evaluate(
    metrics=["perplexity", "forget_quality", "model_utility"]
)

print(f"Forget Quality: {results['forget_quality']:.4f}")
print(f"Model Utility: {results['model_utility']:.4f}")
```

## 📚 Supported Methods

| Method | Description | Reference Model | Best For |
|--------|-------------|----------------|----------|
| **RMU** | Representation Misdirection for Unlearning | Required | Safety-critical applications, robust forgetting |
| **GradDiff** | Gradient Difference (ascent on forget, descent on retain) | Optional | Balanced forget/retain trade-off |
| **DPO** | Direct Preference Optimization | Required | Preference-based unlearning with alternate answers |
| **SimNPO** | Simple Negative Preference Optimization | Not required | Stable unlearning without reference model |
| **GradAscent** | Gradient Ascent on forget set | Not required | Simple baseline, quick experiments |

### Method Selection Guide

```python
# For safety-critical unlearning (e.g., removing hazardous knowledge)
unlearner = Unlearning(method="rmu", model="model_name", adaptive=True)

# For balanced forgetting with good retain data
unlearner = Unlearning(method="grad_diff", model="model_name", 
                       gamma=1.0, alpha=1.0)

# When you have alternate acceptable answers
unlearner = Unlearning(method="dpo", model="model_name", beta=1.0)

# For simple, stable unlearning
unlearner = Unlearning(method="simnpo", model="model_name")

# Quick baseline for experiments
unlearner = Unlearning(method="grad_ascent", model="model_name")
```

## 📖 Detailed Examples

### Example 1: RMU with Adaptive Steering

```python
from unlearun import Unlearning

# RMU is the most robust method for safety-critical unlearning
unlearner = Unlearning(
    method="rmu",
    model="gpt2",
    output_dir="./rmu_model",
    # RMU-specific parameters
    steering_coeff=1.0,  # Steering strength
    target_layer=8,      # Which transformer layer to steer
    adaptive=True        # Use adaptive coefficient (recommended)
)

# Load data from JSON files
unlearner.load_data(
    forget_data="forget_set.json",
    retain_data="retain_set.json",
    max_length=128
)

# Configure training
unlearner.run(
    batch_size=4,
    learning_rate=1e-5,
    num_epochs=3,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    logging_steps=10
)

# Comprehensive evaluation
results = unlearner.evaluate(
    metrics=[
        "perplexity",
        "forget_quality", 
        "model_utility",
        "rouge",
        "verbatim_memorization",
        "mia"
    ]
)
```

### Example 2: Gradient Difference with KL Regularization

```python
from unlearun import Unlearning

# GradDiff with KL divergence for smoother retain preservation
unlearner = Unlearning(
    method="grad_diff",
    model="gpt2-medium",
    output_dir="./graddiff_model",
    # GradDiff-specific parameters
    gamma=1.0,              # Weight for forget loss
    alpha=1.0,              # Weight for retain loss
    retain_loss_type="KL"   # Use KL divergence (requires ref model)
)

unlearner.load_data(
    forget_data="forget.json",
    retain_data="retain.json"
)

unlearner.run(
    batch_size=2,
    learning_rate=5e-5,
    num_epochs=5
)
```

### Example 3: Loading from HuggingFace Dataset

```python
from datasets import load_dataset
from unlearun import Unlearning

# Load from HuggingFace Hub
forget_dataset = load_dataset("your_username/forget_dataset", split="train")
retain_dataset = load_dataset("your_username/retain_dataset", split="train")

unlearner = Unlearning(
    method="simnpo",
    model="meta-llama/Llama-2-7b-hf",
    output_dir="./unlearned_llama"
)

unlearner.load_data(
    forget_data=forget_dataset,
    retain_data=retain_dataset,
    question_key="prompt",  # Specify your column names
    answer_key="completion",
    max_length=512
)

unlearner.run(batch_size=1, num_epochs=3)
```

### Example 4: Custom Evaluation

```python
from unlearun import Unlearning
from unlearun.evaluation import (
    compute_perplexity,
    compute_verbatim_memorization,
    compute_mia
)

# After training
unlearner = Unlearning(method="rmu", model="gpt2", output_dir="./model")
unlearner.load_data(forget_data="forget.json", retain_data="retain.json")
unlearner.run(batch_size=2, num_epochs=3)

# Custom evaluation with specific parameters
forget_ppl = compute_perplexity(
    model=unlearner.model,
    dataset=unlearner.forget_dataset,
    tokenizer=unlearner.tokenizer,
    batch_size=4
)

# Check for verbatim memorization
verbatim_score = compute_verbatim_memorization(
    model=unlearner.model,
    forget_dataset=unlearner.forget_dataset,
    tokenizer=unlearner.tokenizer,
    prefix_length=50,
    max_new_tokens=100,
    num_samples=100
)

# Membership inference attack
mia_score = compute_mia(
    model=unlearner.model,
    forget_dataset=unlearner.forget_dataset,
    retain_dataset=unlearner.retain_dataset,
    tokenizer=unlearner.tokenizer,
    batch_size=4
)

print(f"Forget Perplexity: {forget_ppl:.2f}")
print(f"Verbatim Memorization: {verbatim_score:.4f}")
print(f"MIA AUROC: {mia_score:.4f}")
```

## 📊 Evaluation Metrics

The package includes comprehensive evaluation metrics:

### Forget Quality Metrics
- **Perplexity**: Measures how "forgotten" the data is (higher = better)
- **Verbatim Memorization**: ROUGE score between generated and ground truth
- **Knowledge Retention**: QA accuracy on forget topics

### Utility Preservation Metrics
- **Model Utility**: Performance on retain set
- **General Knowledge**: Evaluation on holdout data
- **Task Performance**: Accuracy on downstream tasks

### Privacy Metrics
- **Membership Inference Attack (MIA)**: Resistance to privacy attacks
- **Extraction Attack**: Difficulty of extracting forgotten data

## 🏗️ Project Structure

```
unlearun/
├── unlearun/
│   ├── __init__.py           # Package entry point
│   ├── core.py               # High-level Unlearning class
│   ├── methods/              # Unlearning methods
│   │   ├── grad_ascent.py
│   │   ├── grad_diff.py
│   │   ├── dpo.py
│   │   ├── rmu.py
│   │   └── simnpo.py
│   ├── data/                 # Data handling
│   │   ├── dataset.py
│   │   └── collators.py
│   ├── trainer/              # Custom trainer
│   │   └── trainer.py
│   ├── utils/                # Utilities
│   │   ├── losses.py
│   │   └── helpers.py
│   └── evaluation/           # Evaluation metrics
│       └── metrics.py
├── tests/                    # Test suite
│   └── test_unlearning.py
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🧪 Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=unlearun --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## 📋 Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.30.0
- Datasets ≥ 2.12.0
- Accelerate ≥ 0.20.0

See `requirements.txt` for full dependency list.

## 🎓 Benchmarks

The package is compatible with standard unlearning benchmarks:

- **TOFU** (Task of Fictitious Unlearning for LLMs)
- **WMDP** (Weapons of Mass Destruction Proxy)
- **MUSE** (Machine Unlearning Six-Way Evaluation)

```python
# Example: Evaluate on TOFU benchmark
from datasets import load_dataset

tofu_forget = load_dataset("locuslab/TOFU", "forget01", split="train")
tofu_retain = load_dataset("locuslab/TOFU", "retain99", split="train")

unlearner = Unlearning(method="rmu", model="phi-1.5")
unlearner.load_data(forget_data=tofu_forget, retain_data=tofu_retain)
unlearner.run(batch_size=2, num_epochs=3)

results = unlearner.evaluate()
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Format code (`black unlearun/ tests/`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/shashuat/unlearun.git
cd unlearun
pip install -e ".[dev]"
pre-commit install  # Optional: for automatic formatting
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use Unlearun in your research, please cite:

```bibtex
@software{unlearun2025,
  title = {Unlearun: Machine Unlearning for Fine-tuned LLMs},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/shashuat/unlearun},
  version = {0.1.0}
}
```

### Key References

This package implements methods from:

```bibtex
@inproceedings{li2024wmdp,
  title={The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning},
  author={Li, Nathaniel and Pan, Alexander and others},
  booktitle={ICML},
  year={2024}
}

@inproceedings{rafailov2023dpo,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and others},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{maini2024tofu,
  title={TOFU: A Task of Fictitious Unlearning for LLMs},
  author={Maini, Pratyush and Feng, Zhili and others},
  booktitle={COLM},
  year={2024}
}
```

## 🙏 Acknowledgments

- Built on [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Inspired by research from CMU, Stanford, and other leading institutions
- Thanks to the machine unlearning research community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/shashuat/unlearun/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shashuat/unlearun/discussions)
- **Email**: your.email@example.com

## 🔗 Links

- **Documentation**: [Full Documentation](https://unlearun.readthedocs.io)
- **PyPI**: [Package on PyPI](https://pypi.org/project/unlearun/)
- **Paper**: [arXiv](https://arxiv.org/abs/xxxx.xxxxx) (coming soon)
- **WMDP Benchmark**: https://www.wmdp.ai/
- **TOFU Benchmark**: https://github.com/locuslab/tofu

---

**Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: October 2025

Made with ❤️ for AI Safety and Privacy