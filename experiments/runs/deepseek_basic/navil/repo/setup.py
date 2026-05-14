from setuptools import setup, find_packages

setup(
    name='navil',
    version='1.0.0',
    description='NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints',
    author='NaViL Reproduction',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'pillow>=9.0.0',
        'pyyaml>=6.0',
    ],
    python_requires='>=3.9',
)
