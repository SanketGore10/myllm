"""
Shared fixtures for pytest.

Provides common test fixtures including:
- Mock llama.cpp models
- Test databases
- Temporary model directories
- FastAPI test clients
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Generator, AsyncGenerator

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.storage.database import init_database


# ============================================================================
# Event Loop Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock llama.cpp
# ============================================================================

@pytest.fixture
def mock_llama_model():
    """
    Mock llama.cpp Llama model for testing without actual models.
    
    Returns a mock that simulates model loading and generation.
    """
    with patch('app.engine.llama_cpp.Llama') as mock_llama:
        # Configure mock for successful loading
        mock_instance = MagicMock()
        mock_llama.return_value = mock_instance
        
        # Mock create_completion for generation
        mock_instance.create_completion.return_value = {
            "choices": [{
                "text": "This is a test response from the mocked model.",
                "finish_reason": "stop"
            }]
        }
        
        # Mock streaming completion
        def streaming_response(*args, **kwargs):
            tokens = ["This", " is", " a", " test", " response", "."]
            for token in tokens:
                yield {
                    "choices": [{
                        "text": token,
                        "finish_reason": None
                    }]
                }
        
        mock_instance.create_completion.side_effect = lambda **kwargs: (
            streaming_response() if kwargs.get("stream") else 
            {"choices": [{"text": "Test response"}]}
        )
        
        # Mock embed
        mock_instance.embed.return_value = [0.1] * 384  # 384-dim embedding
        
        # Mock tokenize
        mock_instance.tokenize.return_value = [1, 2, 3, 4, 5]
        
        yield mock_instance


# ============================================================================
# Test Database
# ============================================================================

@pytest.fixture
async def test_db():
    """
    Create in-memory SQLite database for testing.
    
    Automatically initializes tables and cleans up after test.
    """
    # Use in-memory database
    test_settings = Settings(db_path=":memory:")
    
    # Patch settings to use test database
    with patch('app.storage.database.get_settings', return_value=test_settings):
        await init_database()
        yield
        # Cleanup handled by in-memory DB


# ============================================================================
# Test Model Directory
# ============================================================================

@pytest.fixture
def test_models_dir(tmp_path: Path) -> Path:
    """
    Create temporary models directory with test model configurations.
    
    Args:
        tmp_path: pytest's tmp_path fixture
    
    Returns:
        Path to temporary models directory
    """
    models_dir = tmp_path / "models_data"
    models_dir.mkdir()
    
    # Create test model 1: llama family
    test_model_1 = models_dir / "test-llama"
    test_model_1.mkdir()
    (test_model_1 / "model.gguf").write_bytes(b"fake gguf data")
    (test_model_1 / "config.json").write_text(json.dumps({
        "name": "test-llama",
        "family": "llama",
        "quantization": "Q4_K_M",
        "context_size": 2048,
        "template": "llama",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }))
    
    # Create test model 2: phi family
    test_model_2 = models_dir / "test-phi"
    test_model_2.mkdir()
    (test_model_2 / "model.gguf").write_bytes(b"fake gguf data")
    (test_model_2 / "config.json").write_text(json.dumps({
        "name": "test-phi",
        "family": "phi",
        "quantization": "Q4_K_M",
        "context_size": 2048,
        "template": "phi",
        "parameters": {
            "temperature": 0.7,
           "top_p": 0.9
        }
    }))
    
    return models_dir


# ============================================================================
# FastAPI Test Client
# ============================================================================

@pytest.fixture
def test_client(test_models_dir: Path, mock_llama_model):
    """
    Create FastAPI TestClient with test configuration.
    
    Args:
        test_models_dir: Temporary models directory
        mock_llama_model: Mocked llama.cpp model
    
    Returns:
        TestClient instance for API testing
    """
    # Patch settings to use test models directory
    with patch('app.core.config.get_settings') as mock_settings:
        test_settings = Settings(
            models_dir=test_models_dir,
            db_path=":memory:"
        )
        mock_settings.return_value = test_settings
        
        # Import app after settings are patched
        from app.main import app
        
        client = TestClient(app)
        yield client


# ============================================================================
# Sample Test Data
# ============================================================================

@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with something?"},
    ]


@pytest.fixture
def sample_inference_options():
    """Sample inference options for testing."""
    return {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 256,
    }
