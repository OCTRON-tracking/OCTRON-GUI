"""Fixtures and configurations shared by the entire test suite.

Follows movement's structure at https://github.com/neuroinformatics-unit/movement/tree/main/tests
"""

import numpy as np
import pytest


# define other fixtures shared by the whole suite here
# Example:
@pytest.fixture(scope="session")
def rng():
    """Return a random number generator with a fixed seed."""
    return np.random.default_rng(seed=42)
