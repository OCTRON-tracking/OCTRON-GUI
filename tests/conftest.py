"""Fixtures and configurations shared by the entire test suite.

Follows movement's structure at https://github.com/neuroinformatics-unit/movement/tree/main/tests
"""

import pytest


@pytest.fixture
def touch_file(tmp_path):
    """Return a factory that creates an empty file under ``tmp_path``.

    Parent directories are created as needed. Handy for the throwaway
    ``.mp4`` / ``.pt`` files that only need to exist to get past a
    path-validation check.
    """

    def _touch(relpath):
        p = tmp_path / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        return p

    return _touch
