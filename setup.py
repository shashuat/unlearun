#!/usr/bin/env python
"""Setup script for unlearun package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="unlearun",
    version="0.1.0",
    author="Shashwat",
    author_email="shashmyweb@gmail.com",
    description="Machine unlearning for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shashuat/unlearun",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "tqdm>=4.65.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
        ],
    },
    keywords="machine-learning deep-learning unlearning llm transformers pytorch",
)