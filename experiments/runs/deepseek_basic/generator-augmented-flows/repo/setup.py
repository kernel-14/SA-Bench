from setuptools import setup, find_packages

setup(
    name="consistency_models",
    version="1.0.0",
    description="Improving Consistency Models with Generator-Augmented Flows",
    author="Paper Reproduction",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "numpy",
        "scipy",
    ],
    python_requires=">=3.8",
)
