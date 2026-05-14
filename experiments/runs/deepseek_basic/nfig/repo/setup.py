from setuptools import setup, find_packages

setup(
    name="nfig",
    version="0.1.0",
    description="NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering",
    author="NFIG Reproduction",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
)
