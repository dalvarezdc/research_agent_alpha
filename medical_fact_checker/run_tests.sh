#!/bin/bash
# Run pytest tests for medical fact checker

echo "================================"
echo "Medical Fact Checker - Test Suite"
echo "================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed"
    echo "Install with: pip install -r requirements-test.txt"
    exit 1
fi

# Run tests with different options based on arguments
case "$1" in
    "quick")
        echo "Running quick tests (excluding slow and integration tests)..."
        pytest test_medical_fact_checker.py -v -m "not slow and not integration"
        ;;
    "integration")
        echo "Running integration tests..."
        pytest test_medical_fact_checker.py -v -m "integration"
        ;;
    "coverage")
        echo "Running tests with coverage report..."
        pytest test_medical_fact_checker.py -v --cov=. --cov-report=html --cov-report=term
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    "verbose")
        echo "Running tests with verbose output..."
        pytest test_medical_fact_checker.py -vv -s
        ;;
    "failed")
        echo "Running only previously failed tests..."
        pytest test_medical_fact_checker.py -v --lf
        ;;
    *)
        echo "Running all tests..."
        pytest test_medical_fact_checker.py -v
        ;;
esac

exit_code=$?

echo ""
echo "================================"
if [ $exit_code -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Some tests failed (exit code: $exit_code)"
fi
echo "================================"

exit $exit_code
