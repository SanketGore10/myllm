"""Unit tests for app/models/catalog.py - Model catalog."""

import pytest

from app.models.catalog import (
    MODELS_CATALOG,
    get_model_from_catalog,
    list_catalog_models,
    is_model_in_catalog,
)


class TestModelCatalog:
    """Test model catalog functions."""
    
    def test_catalog_has_models(self):
        """Test catalog contains expected models."""
        assert len(MODELS_CATALOG) > 0
        
        # Check for key models
        expected_models = ["tinyllama-1.1b", "phi-2", "llama3-8b"]
        for model in expected_models:
            assert model in MODELS_CATALOG
    
    def test_catalog_models_have_required_fields(self):
        """Test all catalog models have required fields."""
        required_fields = [
            "repo_id", "filename", "family", "template",
            "context_size", "description", "size_mb"
        ]
        
        for model_name, model_data in MODELS_CATALOG.items():
            for field in required_fields:
                assert field in model_data, f"{model_name} missing {field}"
    
    def test_get_model_from_catalog_success(self):
        """Test get_model_from_catalog returns model data."""
        model_data = get_model_from_catalog("tinyllama-1.1b")
        
        assert model_data is not None
        assert model_data["family"] == "llama"
        assert model_data["template"] == "llama"
    
    def test_get_model_from_catalog_not_found(self):
        """Test get_model_from_catalog returns None for unknown model."""
        model_data = get_model_from_catalog("nonexistent-model")
        
        assert model_data is None
    
    def test_list_catalog_models(self):
        """Test list_catalog_models returns all models."""
        models = list_catalog_models()
        
        assert len(models) == len(MODELS_CATALOG)
        
        # Each model should have name field
        for model in models:
            assert "name" in model
            assert model["name"] in MODELS_CATALOG
    
    def test_is_model_in_catalog(self):
        """Test is_model_in_catalog checks existence."""
        assert is_model_in_catalog("tinyllama-1.1b") is True
        assert is_model_in_catalog("phi-2") is True
        assert is_model_in_catalog("nonexistent") is False
    
    def test_template_matches_family(self):
        """Test that template matches family for consistency."""
        # Template should match family for correct prompt formatting
        model_data = get_model_from_catalog("tinyllama-1.1b")
        assert model_data["family"] == model_data["template"] == "llama"
        
        model_data = get_model_from_catalog("phi-2")
        assert model_data["family"] == model_data["template"] == "phi"
