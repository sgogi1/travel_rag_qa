"""Run all tests and generate coverage report."""

import pytest
import sys
import os

if __name__ == "__main__":
    # Run all tests with coverage
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--cov=retrieval",
        "--cov=indexing",
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-fail-under=100",
        "-x"  # Stop on first failure
    ])
    sys.exit(exit_code)

