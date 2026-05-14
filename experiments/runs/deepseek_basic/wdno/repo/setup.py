from setuptools import setup, find_packages

setup(
    name='wdno',
    version='0.1.0',
    description='Wavelet Diffusion Neural Operator (WDNO)',
    author='WDNO Reproduction',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'tqdm>=4.60.0',
        'h5py>=3.0.0',
        'pytorch-wavelets>=1.3.0',
        'ptwt>=0.1.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
    ],
    python_requires='>=3.8',
)
