"""Setup."""

from setuptools import setup
from setuptools import find_namespace_packages


setup(name='discs',
      packages=find_namespace_packages(),
      install_requires=[
          'ml_collections',
          'numpy',
          'matplotlib',
          'tqdm',
          'tensorflow',
          'networkx'
      ]
)
