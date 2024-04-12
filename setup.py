from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


download_url = (
    f"https://github.com/pfeinsper/drone-swarm-search/archive/refs/tags/v{{VERSION_PLACEHOLDER}}.tar.gz"
)
setup(
    name="DSSE",
    version="{{VERSION_PLACEHOLDER}}",
    author="Luis Filipe Carrete, Manuel Castanares, Enrico Damiani, Leonardo Malta, Joras Oliveira, Ricardo Ribeiro Rodrigues, Renato Lafrachi Falcao, Pedro Andrade, Fabricio Barth",
    description="An environment to train drones to search and find a shipwrecked person lost in the ocean using reinforcement learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pfeinsper/drone-swarm-search",
    license="MIT",
    keywords=["Reinforcement Learning", "AI", "SAR", "Multi Agent"],
    download_url=download_url,
    packages=find_packages(),
    include_package_data=True,
    package_data={"DSSE": ["environment/imgs/*.png"]},
    install_requires=[
        "numpy",
        "gymnasium",
        "pygame",
        "pettingzoo",
        "matplotlib",
        "numba",
    ],
)
