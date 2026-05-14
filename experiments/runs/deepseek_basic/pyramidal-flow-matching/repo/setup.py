from setuptools import setup, find_packages

setup(
    name="pyramidal-flow-matching",
    version="1.0.0",
    description="Pyramidal Flow Matching for Efficient Video Generative Modeling",
    author="Pyramidal Flow Team",
    url="https://pyramid-flow.github.io",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
