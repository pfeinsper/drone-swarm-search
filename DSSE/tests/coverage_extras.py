"""Helper for tests that need the coverage extras.

Locally the tests are skipped when an extra is missing, so a plain
`pip install DSSE` can still run the core tests. In CI (where GitHub Actions
sets CI=true) a missing extra is an error, so the tests cannot pass by being
skipped.
"""

import importlib
import os

import pytest

COVERAGE_MODULES = ("h5py", "xarray", "copernicusmarine", "dotenv", "opendrift")


def require_coverage_extras():
    """Skip the calling test module if an extra is missing, or fail in CI."""
    for name in COVERAGE_MODULES:
        if os.environ.get("CI", "").lower() == "true":
            importlib.import_module(name)
        else:
            pytest.importorskip(name)
