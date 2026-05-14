from setuptools import setup, find_packages

setup(
    name="conformal-bayesian-quadrature",
    version="0.1.0",
    description="Conformal Prediction as Bayesian Quadrature",
    author="Reproduction",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)
