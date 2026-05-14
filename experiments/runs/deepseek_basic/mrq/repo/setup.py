from setuptools import setup, find_packages

setup(
    name="mrq",
    version="0.1.0",
    description="MR.Q: Towards General-Purpose Model-Free RL",
    author="Research Reproduction",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
    ],
    python_requires=">=3.8",
)
