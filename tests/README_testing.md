# Testing Guide for Khmer Name OCR Corrector

This document provides instructions for running tests to verify the functionality of the Khmer Name OCR Corrector system.

## Overview of Testing Suite

The testing suite includes:

1. **Utility Tests** - Test the core functionality of the name matching utilities
2. **API Tests** - Test the FastAPI endpoints
3. **Streamlit App Tests** - Test the Streamlit web application
4. **Docker Tests** - Test the Docker setup

## Prerequisites

Before running tests, ensure you have installed all the required dependencies:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov requests docker
```

## Running All Tests

To run all tests, use the provided test runner script:

```bash
python run_tests.py
```

This will execute all test categories and provide a summary of results.

## Running Specific Tests

To run only a specific test category:

```bash
python run_tests.py --test test_utils  # Run only utility tests
python run_tests.py --test test_api    # Run only API tests
python run_tests.py --test test_app    # Run only Streamlit app tests
python run_tests.py --test test_docker # Run only Docker tests
```

Or you can run individual test files directly:

```bash
python -m tests.test_utils
python -m tests.test_api
python -m tests.test_app
python -m tests.test_docker
```

## Test Data

The tests use a small sample dataset stored in `sample.csv`. The format is:

```
latin_name,khmer_name
songhieng,សុងហៀង
dara,ដារា
sok,សុខ
vichet,វិចិត្រ
```

## Test Categories in Detail

### Utility Tests (`tests/test_utils.py`)

Tests the core functionality:
- KhmerNameMatcher initialization
- Khmer text cleaning 
- Latin text normalization
- Name matching with both Khmer and Latin text
- Batch matching functionality

### API Tests (`tests/test_api.py`)

Tests the FastAPI endpoints:
- Root endpoint
- Info endpoint
- Match endpoint
- Batch-match endpoint

### Streamlit App Tests (`tests/test_app.py`)

Tests the Streamlit web application:
- Application startup
- Basic UI functionality

### Docker Tests (`tests/test_docker.py`)

Tests the Docker setup:
- Image building
- Container running
- API functionality within Docker

## Troubleshooting

### API Tests

If API tests fail with connection errors:
- Ensure no other service is using port 8000
- Check that FastAPI and its dependencies are installed

### Streamlit App Tests

If Streamlit tests fail:
- Ensure no other service is using the selected port
- Check that Streamlit and its dependencies are installed

### Docker Tests

If Docker tests fail:
- Ensure Docker is installed and running
- Ensure your user has permissions to use Docker
- Check that the docker-py library is installed

## Continuous Integration

These tests can be integrated into CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Run Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov requests docker
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Run tests
      run: |
        python run_tests.py --continue-on-error
```

## Extending Tests

To add new tests:

1. Create a new test function in the appropriate test file
2. Follow the naming convention `test_*` for test functions
3. Use assertions to verify expected behavior
4. Update the test runner if necessary

## Code Coverage

To generate a code coverage report:

```bash
pytest --cov=. --cov-report=html
```

This will generate an HTML report in the `htmlcov` directory. 