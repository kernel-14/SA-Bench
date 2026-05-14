"""Setup script for SAM 2."""

from setuptools import setup, find_packages

setup(
    name="sam2",
    version="1.0.0",
    description="SAM 2: Segment Anything in Images and Videos",
    author="Meta FAIR",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "Pillow",
        "scipy",
        "pycocotools",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
)
