"""Setup script for the gated_attention package."""

from setuptools import setup, find_packages

setup(
    name="gated-attention-llm",
    version="0.1.0",
    description="Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free",
    author="Paper Reproduction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
