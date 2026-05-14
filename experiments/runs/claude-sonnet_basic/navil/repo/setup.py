from setuptools import setup, find_packages

setup(
    name="navil",
    version="1.0.0",
    description="NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.40.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
