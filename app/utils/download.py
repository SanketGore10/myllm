"""
Download utilities for fetching models from Hugging Face.

Handles file downloads with progress tracking and verification.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Callable

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DownloadError(Exception):
    """Exception raised when download fails."""
    pass


def download_model_from_hf(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Download a model file from Hugging Face Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        filename: Name of the file to download
        local_dir: Local directory to save the file
        token: HuggingFace API token (optional, for private repos)
        progress_callback: Callback function(progress: float) for progress updates (0-100)
    
    Returns:
        Path to the downloaded file
    
    Raises:
        DownloadError: If download fails
    """
    try:
        logger.info(f"Downloading {filename} from {repo_id}")
        
        # Ensure local directory exists
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download file from HuggingFace
        # Note: hf_hub_download handles progress internally, but we can't easily hook into it
        # For now, we'll just download and report completion
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Copy actual file, not symlink
            token=token,
        )
        
        logger.info(f"Download complete: {file_path}")
        
        # Report 100% completion
        if progress_callback:
            progress_callback(100.0)
        
        return Path(file_path)
    
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            error_msg = f"Model file not found: {repo_id}/{filename}"
        elif e.response.status_code == 401:
            error_msg = f"Authentication required. Please set HF_TOKEN environment variable."
        else:
            error_msg = f"HTTP error downloading model: {e}"
        
        logger.error(error_msg)
        raise DownloadError(error_msg)
    
    except Exception as e:
        error_msg = f"Failed to download model: {e}"
        logger.error(error_msg)
        raise DownloadError(error_msg)


def verify_file_integrity(
    file_path: Path,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
) -> bool:
    """
    Verify downloaded file integrity.
    
    Args:
        file_path: Path to the file
        expected_size: Expected file size in bytes (optional)
        expected_sha256: Expected SHA256 hash (optional)
    
    Returns:
        True if file is valid
    
    Raises:
        ValueError: If verification fails
    """
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    
    # Check file size
    actual_size = file_path.stat().st_size
    
    if expected_size and actual_size != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size} bytes, "
            f"got {actual_size} bytes"
        )
    
    # Check SHA256 hash (if provided)
    if expected_sha256:
        logger.info("Verifying file hash...")
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_hash = sha256_hash.hexdigest()
        
        if actual_hash != expected_sha256:
            raise ValueError(
                f"SHA256 hash mismatch: expected {expected_sha256}, "
                f"got {actual_hash}"
            )
        
        logger.info("File hash verified successfully")
    
    logger.info(f"File verification passed: {file_path}")
    return True


def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in bytes
    """
    return file_path.stat().st_size


def format_size(size_bytes: int) -> str:
    """
    Format byte size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def check_disk_space(directory: Path, required_bytes: int) -> bool:
    """
    Check if there's enough disk space available.
    
    Args:
        directory: Directory to check
        required_bytes: Required space in bytes
    
    Returns:
        True if enough space is available
    """
    import shutil
    
    stat = shutil.disk_usage(directory)
    available_bytes = stat.free
    
    logger.debug(
        f"Disk space check: {format_size(available_bytes)} available, "
        f"{format_size(required_bytes)} required"
    )
    
    return available_bytes >= required_bytes
