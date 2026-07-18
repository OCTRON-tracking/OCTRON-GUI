"""Fixtures and configurations shared by the entire test suite.

Follows movement's structure at https://github.com/neuroinformatics-unit/movement/tree/main/tests
"""

from pathlib import Path

import numpy as np
import pytest

# define `fixtures`` module path
# by searching relative to this file
# (we assume the structure is always
#  tests.fixtures.<module_name>)
fixtures_dir = Path(__file__).parent / "fixtures"
pytest_plugins = [
    f"tests.fixtures.{fixture.stem}"
    for fixture in fixtures_dir.glob("*.py")
    if "__" not in fixture.name
]

# define other fixtures shared by the whole suite here
@pytest.fixture(scope="session")
def rng():
    """Return a random number generator with a fixed seed."""
    return np.random.default_rng(seed=42)