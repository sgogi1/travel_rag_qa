#!/bin/bash
# Comprehensive test runner script

echo "=========================================="
echo "Travel Agency RAG System - Test Suite"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

echo "Running unit tests..."
pytest tests/unit/ -v --tb=short
UNIT_EXIT=$?

echo ""
echo "Running integration tests..."
pytest tests/integration/ -v --tb=short
INTEGRATION_EXIT=$?

echo ""
echo "Running end-to-end tests..."
pytest tests/e2e/ -v --tb=short
E2E_EXIT=$?

echo ""
echo "=========================================="
echo "Generating coverage report..."
echo "=========================================="
pytest tests/ --cov=retrieval --cov=indexing --cov=app \
    --cov-report=term-missing --cov-report=html --cov-report=xml \
    --tb=short

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Unit tests: $([ $UNIT_EXIT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "Integration tests: $([ $INTEGRATION_EXIT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "E2E tests: $([ $E2E_EXIT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo ""
echo "Coverage report: htmlcov/index.html"
echo "=========================================="

# Exit with error if any test suite failed
if [ $UNIT_EXIT -ne 0 ] || [ $INTEGRATION_EXIT -ne 0 ] || [ $E2E_EXIT -ne 0 ]; then
    exit 1
fi

exit 0

