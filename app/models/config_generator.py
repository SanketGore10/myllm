"""
Configuration generator for downloaded models.

Auto-generates config.json from model metadata.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


def detect_quantization(filename: str) -> str:
    """
    Detect quantization type from filename.
    
    Args:
        filename: Model filename (e.g., "model.Q4_K_M.gguf")
    
    Returns:
        Quantization type (e.g., "Q4_K_M")
    """
    # Common quantization patterns
    patterns = [
        r'Q\d+_[KM0](?:_[SMLXYZ])?',  # Q4_K_M, Q5_K_S, Q8_0, etc.
        r'F16',
        r'F32',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(0).upper()
    
    # Default if not found
    logger.warning(f"Could not detect quantization from filename: {filename}")
    return "unknown"


def generate_config(
    model_name: str,
    family: str,
    template: str,
    context_size: int,
    model_file_path: Path,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate config.json for a model.
    
    Args:
        model_name: Model name
        family: Model family (llama, mistral, etc.)
        template: Prompt template (chatml, llama3, etc.)
        context_size: Context window size
        model_file_path: Path to the model file
        description: Model description (optional)
        parameters: Additional parameters (optional)
    
    Returns:
        Config dictionary
    """
    # Detect quantization from filename
    quantization = detect_quantization(model_file_path.name)
    
    # Default parameters
    default_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
    
    # Merge with provided parameters
    if parameters:
        default_params.update(parameters)
    
    config = {
        "name": model_name,
        "family": family,
        "quantization": quantization,
        "context_size": context_size,
        "template": template,
        "parameters": default_params,
    }
    
    if description:
        config["description"] = description
    
    logger.info(f"Generated config for {model_name}: {config}")
    
    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save config to JSON file.
    
    Args:
        config: Config dictionary
        config_path: Path to save config.json
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Config saved to {config_path}")


def create_model_config(
    model_name: str,
    model_dir: Path,
    catalog_metadata: Dict[str, Any],
) -> Path:
    """
    Create complete config.json for a downloaded model.
    
    Args:
        model_name: Model name
        model_dir: Model directory
        catalog_metadata: Metadata from catalog
    
    Returns:
        Path to created config.json
    """
    model_file = model_dir / "model.gguf"
    config_path = model_dir / "config.json"
    
    # Generate config from catalog metadata
    config = generate_config(
        model_name=model_name,
        family=catalog_metadata["family"],
        template=catalog_metadata["template"],
        context_size=catalog_metadata["context_size"],
        model_file_path=model_file,
        description=catalog_metadata.get("description"),
    )
    
    # Save config
    save_config(config, config_path)
    
    return config_path
