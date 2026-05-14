from setuptools import setup, find_packages

setup(
    name="moe-pot",
    version="0.1.0",
    description="MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "h5py",
        "pyyaml",
        "tqdm",
        "matplotlib",
    ],
)
