from setuptools import setup, find_packages
import sys

with open("README.md", 'r') as f:
    long_description = f.read()


setup(name = 'DSSE',
    version = '{{VERSION_PLACEHOLDER}}',
    author = "Luis Filipe Carrete, Manuel Castanares, Enrico Damiani, Leonardo Malta, Joras Oliveira, Ricardo Ribeiro Rodrigues, Renato Laffranchi Falcão, Pedro Andrade, Fabricio Barth",
    description = 'An environment to train drones to search and find a shipwrecked person lost in the ocean using reinforcement learning.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/pfeinsper/drone-swarm-search',
    license = 'MIT',
    keywords = ['Reinforcement Learning', 'AI', 'SAR', 'Multi Agent'],
    download_url = f'https://github.com/pfeinsper/drone-swarm-search/archive/refs/tags/v{{VERSION_PLACEHOLDER}}.tar.gz',
    include_package_data=True, 
    packages = ['DSSE', 'DSSE.core', 'DSSE.core.environment', 'DSSE.core.environment.generator', 'DSSE.core.environment.imgs'],
    install_requires = [
        'numpy',
        'gymnasium',
        'pygame',
        'pettingzoo',
        'matplotlib',
      ],)
