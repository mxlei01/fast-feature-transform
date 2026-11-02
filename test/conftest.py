"""Shared pytest configuration: the opt-in ``scale`` marker."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-scale`` flag.

    Args:
        parser: The pytest argument parser.

    """
    parser.addoption("--run-scale", action="store_true", default=False, help="run the large out-of-core tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``scale``-marked tests unless ``--run-scale`` was passed.

    Args:
        config: The pytest configuration.
        items: The collected test items.

    """
    if config.getoption("--run-scale"):
        return
    skip = pytest.mark.skip(reason="large out-of-core test; pass --run-scale to run it")
    for item in items:
        if "scale" in item.keywords:
            item.add_marker(skip)
