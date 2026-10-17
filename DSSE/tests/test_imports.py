import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from DSSE.tests.coverage_extras import COVERAGE_MODULES, require_coverage_extras

# Runs in a fresh interpreter with the coverage dependencies blocked,
# as after a plain `pip install DSSE`.
CORE_ONLY_SCRIPT = textwrap.dedent(f"""
    import sys
    for name in {list(COVERAGE_MODULES)!r}:
        sys.modules[name] = None

    from DSSE import Actions, CoverageDroneSwarmSearch, DroneSwarmSearch

    env = DroneSwarmSearch(grid_size=20, drone_amount=2, timestep_limit=10)
    observations, info = env.reset()
    actions = {{agent: Actions.SEARCH.value for agent in env.agents}}
    env.step(actions)

    try:
        from DSSE import AdvancedCoverageDroneSwarmSearch
    except ImportError as error:
        assert "DSSE[coverage]" in str(error), error
    else:
        raise AssertionError("expected an ImportError without the coverage extras")
    print("ok")
    """)


def test_core_env_imports_without_coverage_extras():
    result = subprocess.run(
        [sys.executable, "-c", CORE_ONLY_SCRIPT],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_unknown_attribute_raises_attribute_error():
    import DSSE

    with pytest.raises(AttributeError):
        DSSE.NotAnExport


def test_coverage_exports_load_when_installed():
    require_coverage_extras()

    from DSSE import (
        AdvancedCoverageDroneSwarmSearch,
        DynamicParticleSimulation,
        H5DatasetBuilder,
    )

    assert (
        AdvancedCoverageDroneSwarmSearch.__name__ == "AdvancedCoverageDroneSwarmSearch"
    )
    assert DynamicParticleSimulation.__name__ == "DynamicParticleSimulation"
    assert H5DatasetBuilder.__name__ == "H5DatasetBuilder"


def test_star_import_without_coverage_extras():
    script = textwrap.dedent(f"""
        import sys
        for name in {list(COVERAGE_MODULES)!r}:
            sys.modules[name] = None
            
        import inspect
        import DSSE
        inspect.getmembers(DSSE)  # pydoc and IDEs call getattr on every name in dir()
        
        namespace = {{}}
        exec("from DSSE import *", namespace)
        assert "DroneSwarmSearch" in namespace
        assert "AdvancedCoverageDroneSwarmSearch" not in namespace
        print("ok")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
