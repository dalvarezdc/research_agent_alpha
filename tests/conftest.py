#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for medical fact checker tests
"""

import pytest
from pathlib import Path


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


@pytest.fixture(autouse=True, scope="session")
def isolate_test_database(tmp_path_factory):
    """Ensure pytest always uses an isolated temporary SQLite database and never pollutes production data/app.db."""
    import os
    from database.config import reset_engine_cache
    from database.session import reset_initialized_flag

    test_db = tmp_path_factory.mktemp("db") / "test_app.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{test_db}"
    os.environ["DB_PERSISTENCE_ENABLED"] = "true"
    reset_engine_cache()
    reset_initialized_flag()
    yield
    reset_engine_cache()
    reset_initialized_flag()


@pytest.fixture(autouse=True)
def cleanup_outputs(request):
    """Cleanup any test output files after each test"""
    yield
    # Cleanup logic here if needed
