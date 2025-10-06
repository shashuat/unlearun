# unlearun

# unlearun Package Structure

```
unlearun/
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── tests/
│   ├── __init__.py
│   └── test_unlearning.py
└── unlearun/
    ├── __init__.py
    ├── core.py
    ├── methods/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── grad_ascent.py
    │   ├── grad_diff.py
    │   ├── dpo.py
    │   ├── rmu.py
    │   └── simnpo.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py
    │   └── collators.py
    ├── trainer/
    │   ├── __init__.py
    │   └── trainer.py
    ├── utils/
    │   ├── __init__.py
    │   ├── losses.py°°
    │   └── helpers.py
    └── evaluation/
        ├── __init__.py
        └── metrics.py
```