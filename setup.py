from setuptools import setup, find_packages
import sys

with open("README.md", 'r') as f:
    long_description = f.read()


setup(name = 'DSSE',
      version = '0.1.17.14',
      description = 'An environment to train drones to search and find a shipwrecked person lost in the ocean using reinforcement learning.',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      url = 'https://github.com/pfeinsper/drone-swarm-search',
      license = 'MIT',
      #packages = find_packages(),
      #data_files=[('DSSE/core/environment/imgs', ['DSSE/core/environment/imgs/drone.png', 'DSSE/core/environment/imgs/person-swimming.png'])],
      packages = ['DSSE', 'DSSE.core', 'DSSE.core.environment', 'DSSE.core.environment.generator', 'DSSE.core.environment.imgs'],
      install_requires = [
          'numpy',
          'gymnasium',
          'pygame',
          'pettingzoo',
          'matplotlib',
      ],)
