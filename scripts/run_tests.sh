#!/bin/bash
# Run all tests

set -e

echo "Running Unit Tests..."
cd backend
pytest tests/unit/ -v

echo "Running Integration Tests..."
pytest tests/integration/ -v

echo "All tests passed successfully!"
