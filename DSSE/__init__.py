"""DSSE: A Multi-Agent Reinforcement Learning Environment for SAR missions using drone swarms."""

import importlib

from .environment.env import DroneSwarmSearch
from .environment.coverage_env import CoverageDroneSwarmSearch
from .environment.constants import Actions, AdvancedActions

# These need the coverage extras (h5py, opendrift, xarray, copernicusmarine,
# python-dotenv), so they are imported on first use. This keeps
# `from DSSE import DroneSwarmSearch` working with a plain `pip install DSSE`.
_COVERAGE_EXPORTS = {
    "AdvancedCoverageDroneSwarmSearch": ".environment.advanced_coverage_env",
    "DynamicParticleSimulation": ".environment.simulation.dynamic_particle_simulation",
    "H5DatasetBuilder": ".environment.simulation.H5DatasetBuilder",
}


def __getattr__(name):
    if name not in _COVERAGE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(_COVERAGE_EXPORTS[name], __name__)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] == __name__:
            raise
        raise ImportError(
            f"{name} needs the coverage dependencies, which are not installed "
            f"(missing module: {error.name!r}). "
            'Install them with: pip install "DSSE[coverage]"'
        ) from error
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_COVERAGE_EXPORTS))


__all__ = [
    "DroneSwarmSearch",
    "CoverageDroneSwarmSearch",
    "Actions",
    "AdvancedCoverageDroneSwarmSearch",
    "DynamicParticleSimulation",
    "H5DatasetBuilder",
    "AdvancedActions",
]
