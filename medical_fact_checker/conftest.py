#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for medical fact checker tests
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API access"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def outputs_dir(tmp_path_factory):
    """Create temporary outputs directory for tests"""
    output_dir = tmp_path_factory.mktemp("outputs")
    return output_dir


@pytest.fixture(autouse=True)
def cleanup_outputs(request):
    """Cleanup any test output files after each test"""
    yield
    # Cleanup logic here if needed
