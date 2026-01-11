"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check(self, test_client):
        """Test health check returns 200."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Test / root endpoint."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns system info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert data["name"] == "MyLLM"
        assert "version" in data
        assert "status" in data


class TestModelsEndpoint:
    """Test /api/models endpoints."""
    
    def test_list_models(self, test_client):
        """Test GET /api/models returns model list."""
        response = test_client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_get_model_details(self, test_client):
        """Test GET /api/models/{name} returns model details."""
        # First, get list of models
        models_response = test_client.get("/api/models")
        models = models_response.json()["models"]
        
        if models:
            model_name = models[0]["name"]
            
            response = test_client.get(f"/api/models/{model_name}")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["name"] == model_name
            assert "family" in data
            assert "context_size" in data
    
    def test_get_nonexistent_model(self, test_client):
        """Test GET /api/models/{name} returns 404 for unknown model."""
        response = test_client.get("/api/models/nonexistent-model-xyz")
        
        assert response.status_code == 404


# Note: Chat, generate, and embeddings endpoints require model loading
# which is mocked. Full integration tests would be in e2e tests.
