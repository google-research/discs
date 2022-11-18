"""Setup."""

from setuptools import setup


setup(name='discs',
      py_modules=['discs'],
      install_requires=[
          'ml_collections',
          'numpy',
          'matplotlib',
          'tqdm',
          'tensorflow'
      ]
)
