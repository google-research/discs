"""Setup."""

from setuptools import find_namespace_packages
from setuptools import setup


setup(
    name='discs',
    packages=find_namespace_packages(),
    install_requires=[
        'ml_collections',
        'numpy',
        'matplotlib',
        'tqdm',
        'tensorflow',
        'networkx',
        'transformers>=4.6.1',
        'tensorflow_probability',
        'absl-py',
        'clu',
        'flax',
        'optax',
        'python-sat',
        'tensorboard',
        'pickle5',
        'nltk',
    ],
)
