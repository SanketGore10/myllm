"""
Model catalog with pre-configured models available for download.

Contains metadata for popular GGUF models from Hugging Face.
"""

from typing import Dict, Any, Optional, List

# Model catalog with metadata for popular models
MODELS_CATALOG: Dict[str, Dict[str, Any]] = {
    "tinyllama-1.1b": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "family": "llama",
        "template": "llama",  # FIXED: TinyLlama is LLaMA family, not ChatML
        "context_size": 2048,
        "description": "TinyLlama 1.1B Q4_K_M - Small, fast model for testing (~637MB)",
        "size_mb": 637,
    },
    "phi-2": {
        "repo_id": "TheBloke/phi-2-GGUF",
        "filename": "phi-2.Q4_K_M.gguf",
        "family": "phi",
        "template": "phi",  # FIXED: Phi uses Alpaca format, not ChatML
        "context_size": 2048,
        "description": "Microsoft Phi-2 2.7B Q4_K_M - Efficient small model (~1.6GB)",
        "size_mb": 1600,
    },
    "llama3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "family": "llama",
        "template": "llama3",
        "context_size": 8192,
        "description": "Llama 3 8B Instruct Q4_K_M - High quality general purpose (~4.9GB)",
        "size_mb": 4920,
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "family": "mistral",
        "template": "mistral",
        "context_size": 8192,
        "description": "Mistral 7B Instruct v0.2 Q4_K_M - Powerful 7B model (~4.4GB)",
        "size_mb": 4370,
    },
    "qwen-1.8b": {
        "repo_id": "Qwen/Qwen1.5-1.8B-Chat-GGUF",
        "filename": "qwen1_5-1_8b-chat-q4_k_m.gguf",
        "family": "qwen",
        "template": "chatml",
        "context_size": 32768,
        "description": "Qwen 1.5 1.8B Chat Q4_K_M - Long context small model (~1.1GB)",
        "size_mb": 1100,
    },
}


def get_model_from_catalog(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get model metadata from catalog.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Model metadata dict or None if not found
    """
    return MODELS_CATALOG.get(model_name)


def list_catalog_models() -> List[Dict[str, Any]]:
    """
    Get list of all models in catalog with their names.
    
    Returns:
        List of dicts with name and metadata
    """
    return [
        {"name": name, **metadata}
        for name, metadata in MODELS_CATALOG.items()
    ]


def search_catalog(query: str) -> List[Dict[str, Any]]:
    """
    Search catalog by name or description.
    
    Args:
        query: Search query
    
    Returns:
        List of matching models
    """
    query_lower = query.lower()
    results = []
    
    for name, metadata in MODELS_CATALOG.items():
        if (query_lower in name.lower() or 
            query_lower in metadata["description"].lower() or
            query_lower in metadata["family"].lower()):
            results.append({"name": name, **metadata})
    
    return results


def is_model_in_catalog(model_name: str) -> bool:
    """
    Check if model exists in catalog.
    
    Args:
        model_name: Name of the model
    
    Returns:
        True if model is in catalog
    """
    return model_name in MODELS_CATALOG
