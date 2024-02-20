from setuptools import setup
import sys

with open("README.md", 'r') as f:
    long_description = f.read()


setup(name = 'DSSE',
      version = '0.1.17.4',
      description = 'An environment to train drones to search and find a shipwrecked person lost in the ocean using reinforcement learning.',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url = 'https://github.com/pfeinsper/drone-swarm-search',
      license = 'MIT',
      packages = ['DSSE'],
      install_requires = [
          'numpy',
          'gymnasium',
          'pygame',
          'pettingzoo',
          'matplotlib',
      ],)
