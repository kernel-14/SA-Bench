from setuptools import setup, find_packages

setup(
    name="himar",
    version="1.0.0",
    description="Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots",
    author="Hi-MAR Authors",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
)
