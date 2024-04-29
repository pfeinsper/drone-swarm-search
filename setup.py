from setuptools import setup

download_url = f"https://github.com/pfeinsper/drone-swarm-search/archive/refs/tags/v{{VERSION_PLACEHOLDER}}.tar.gz"
setup(
    version="{{VERSION_PLACEHOLDER}}",
    download_url=download_url,
)
