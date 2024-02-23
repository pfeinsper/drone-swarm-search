from setuptools import setup, find_packages
import sys

with open("README.md", 'r') as f:
    long_description = f.read()


setup(name = 'DSSE',
    version = '0.2',
    author = "Luis Filipe Carrete, Manuel Castanares, Enrico Damiani, Leonardo Malta, Joras Oliveira, Ricardo Ribeiro Rodrigues, Renato Lafrachi Falcao, Pedro Andrade, Fabricio Barth",
    description = 'An environment to train drones to search and find a shipwrecked person lost in the ocean using reinforcement learning.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/pfeinsper/drone-swarm-search',
    license = 'MIT',
    keywords = ['Reinforcement Learning', 'AI', 'SAR', 'Multi Agent'],
    download_url = 'https://github.com/pfeinsper/drone-swarm-search/archive/refs/tags/v0.2.0.tar.gz',
    include_package_data=True, 
    packages = ['DSSE', 'DSSE.core', 'DSSE.core.environment', 'DSSE.core.environment.generator', 'DSSE.core.environment.imgs'],
    install_requires = [
        'numpy',
        'gymnasium',
        'pygame',
        'pettingzoo',
        'matplotlib',
      ],)
