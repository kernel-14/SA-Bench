"""Setup script for OLMoE."""

from setuptools import setup, find_packages

setup(
    name="olmoe",
    version="0.1.0",
    description="OLMoE: Open Mixture-of-Experts Language Models",
    author="OLMoE Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
)
