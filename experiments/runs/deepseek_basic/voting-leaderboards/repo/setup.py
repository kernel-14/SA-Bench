from setuptools import setup, find_packages

setup(
    name="voting-leaderboard-attacks",
    version="0.1.0",
    description="Reproduction of 'Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards'",
    author="Reproduction attempt",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
)
