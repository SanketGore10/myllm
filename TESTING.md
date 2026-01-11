# Running Tests

## Installation

Install development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app --cov-report=html
```

View coverage report:

```bash
# Open htmlcov/index.html in browser
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
```

## Test Categories

### Unit Tests
Fast, isolated tests for individual components:

```bash
pytest tests/unit/
```

### Integration Tests
Tests for component interactions and API endpoints:

```bash
pytest tests/integration/
```

### End-to-End Tests
Full workflow tests (requires actual models):

```bash
pytest tests/e2e/
```

## Running Specific Tests

Single file:

```bash
pytest tests/unit/test_templates.py
```

Single test:

```bash
pytest tests/unit/test_templates.py::TestLLaMATemplate::test_single_user_message
```

By marker:

```bash
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m "not slow"    # Skip slow tests
```

## Test Fixtures

Key fixtures in `tests/conftest.py`:

- `mock_llama_model`: Mocked llama.cpp for testing without models
- `test_db`: In-memory SQLite database
- `test_models_dir`: Temporary directory with test models
- `test_client`: FastAPI TestClient for API testing

## Coverage Goals

- **Unit tests**: 80% coverage
- **Integration tests**: All API endpoints
- **E2E tests**: All CLI commands

## CI/CD

Tests run automatically on GitHub Actions for every push/PR.

See `.github/workflows/tests.yml` for configuration.
