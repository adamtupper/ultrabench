"""Test configuration for the training test suite."""

import sys
from unittest.mock import MagicMock


def pytest_configure(config: MagicMock) -> None:
    """Stub out optional training-layer imports unavailable in the test env.

    Args:
        config: The pytest configuration object.
    """
    sys.modules.setdefault("data", MagicMock())
    sys.modules.setdefault("data.user_preprocessing", MagicMock())
