"""Setup script for the SCoRe package."""

from setuptools import setup, find_packages

setup(
    name="score-rl",
    version="0.1.0",
    description="Training Language Models to Self-Correct via Reinforcement Learning (SCoRe)",
    author="SCoRe Reproduction",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
)
