from setuptools import setup, find_packages

setup(
    name="luno",
    version="0.1.0",
    description="Linearization Turns Neural Operators into Function-Valued Gaussian Processes",
    author="LUNO Reproduction",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
    ],
    python_requires=">=3.8",
)
