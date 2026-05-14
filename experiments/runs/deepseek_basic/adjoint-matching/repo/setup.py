"""Setup for adjoint_matching package."""

from setuptools import setup, find_packages

setup(
    name="adjoint_matching",
    version="0.1.0",
    description="Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC",
    author="Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
)
