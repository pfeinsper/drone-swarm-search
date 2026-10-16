"""DSSE: A Multi-Agent Reinforcement Learning Environment for SAR missions using drone swarms."""

import importlib
import importlib.util
from typing import TYPE_CHECKING

from .environment.env import DroneSwarmSearch
from .environment.coverage_env import CoverageDroneSwarmSearch
from .environment.constants import Actions, AdvancedActions

# The coverage exports need the coverage extras (h5py, opendrift, xarray,
# copernicusmarine, python-dotenv), so they are imported on first use by
# __getattr__ below. This keeps `from DSSE import DroneSwarmSearch` working with
# a plain `pip install DSSE`. Type checkers and IDEs do not run __getattr__, so
# the imports are repeated here for them only.
if TYPE_CHECKING:
    from .environment.advanced_coverage_env import AdvancedCoverageDroneSwarmSearch
    from .environment.simulation.dynamic_particle_simulation import (
        DynamicParticleSimulation,
    )
    from .environment.simulation.H5DatasetBuilder import H5DatasetBuilder

_COVERAGE_DEPENDENCIES = ("h5py", "xarray", "copernicusmarine", "dotenv", "opendrift")

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


def _coverage_installed():
    return all(
        importlib.util.find_spec(module) is not None
        for module in _COVERAGE_DEPENDENCIES
    )


__all__ = [
    "DroneSwarmSearch",
    "CoverageDroneSwarmSearch",
    "Actions",
    "AdvancedActions",
]
# Only advertise the coverage exports to `from DSSE import *` when they can be
# imported, so a star import works with a plain `pip install DSSE`.
if _coverage_installed():
    __all__ += list(_COVERAGE_EXPORTS)
