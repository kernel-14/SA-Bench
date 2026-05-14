from setuptools import setup, find_packages

setup(
    name="ca2-vdm",
    version="1.0.0",
    description="Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "einops>=0.6.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
    ],
)
