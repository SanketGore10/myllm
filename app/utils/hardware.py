"""
Hardware detection utilities.

Detects GPU availability (CUDA, Metal, ROCm) and provides optimal configuration
suggestions for llama.cpp based on available hardware.
"""

import os
import platform
import subprocess
from typing import Optional, Tuple
import psutil

from app.utils.logging import get_logger

logger = get_logger(__name__)


class HardwareInfo:
    """Hardware information container."""
    
    def __init__(
        self,
        gpu_type: Optional[str],
        gpu_memory_mb: int,
        cpu_cores: int,
        total_ram_mb: int,
        suggested_n_gpu_layers: int,
        suggested_threads: int,
    ):
        self.gpu_type = gpu_type  # "cuda", "metal", "rocm", or None
        self.gpu_memory_mb = gpu_memory_mb
        self.cpu_cores = cpu_cores
        self.total_ram_mb = total_ram_mb
        self.suggested_n_gpu_layers = suggested_n_gpu_layers
        self.suggested_threads = suggested_threads
    
    def __repr__(self) -> str:
        return (
            f"HardwareInfo(gpu={self.gpu_type}, "
            f"gpu_mem={self.gpu_memory_mb}MB, "
            f"cpu_cores={self.cpu_cores}, "
            f"ram={self.total_ram_mb}MB)"
        )


def detect_gpu() -> Optional[str]:
    """
    Detect GPU availability and type.
    
    Returns:
        "cuda", "metal", "rocm", or None
    """
    # Check for NVIDIA CUDA
    if _check_cuda():
        logger.info("Detected CUDA GPU")
        return "cuda"
    
    # Check for Apple Metal
    if _check_metal():
        logger.info("Detected Metal GPU (Apple Silicon)")
        return "metal"
    
    # Check for AMD ROCm
    if _check_rocm():
        logger.info("Detected ROCm GPU (AMD)")
        return "rocm"
    
    logger.info("No GPU detected, will use CPU")
    return None


def _check_cuda() -> bool:
    """Check if NVIDIA CUDA is available."""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_metal() -> bool:
    """Check if Apple Metal is available."""
    # Metal is available on macOS with Apple Silicon
    if platform.system() == "Darwin":
        # Check if running on Apple Silicon
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "Apple" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    return False


def _check_rocm() -> bool:
    """Check if AMD ROCm is available."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_gpu_memory() -> int:
    """
    Get GPU memory in MB.
    
    Returns:
        GPU memory in MB, or 0 if not available
    """
    gpu_type = detect_gpu()
    
    if gpu_type == "cuda":
        return _get_cuda_memory()
    elif gpu_type == "metal":
        return _get_metal_memory()
    elif gpu_type == "rocm":
        return _get_rocm_memory()
    
    return 0


def _get_cuda_memory() -> int:
    """Get NVIDIA GPU memory via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Returns memory in MB
            return int(result.stdout.strip().split('\n')[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def _get_metal_memory() -> int:
    """Get Apple GPU memory (unified memory)."""
    try:
        # On macOS, GPU shares system RAM
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Convert bytes to MB
            return int(result.stdout.strip()) // (1024 * 1024)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def _get_rocm_memory() -> int:
    """Get AMD GPU memory via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse output (implementation depends on rocm-smi output format)
        # This is a simplified version
        if result.returncode == 0:
            # Parse memory from output
            # Format varies, this is a placeholder
            return 8192  # Default assumption
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def get_cpu_info() -> Tuple[int, int]:
    """
    Get CPU information.
    
    Returns:
        (cpu_cores, total_ram_mb)
    """
    cpu_cores = psutil.cpu_count(logical=False) or 4
    total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
    
    return cpu_cores, total_ram_mb


def suggest_n_gpu_layers(model_size_mb: int, gpu_memory_mb: int) -> int:
    """
    Suggest optimal number of GPU layers based on model size and GPU memory.
    
    Args:
        model_size_mb: Model file size in MB (rough approximation)
        gpu_memory_mb: Available GPU memory in MB
    
    Returns:
        Suggested n_gpu_layers (-1 for all layers, 0 for CPU-only)
    """
    if gpu_memory_mb == 0:
        # No GPU
        return 0
    
    # Rule of thumb: model needs ~1.2x its size in VRAM for full GPU offloading
    # Plus overhead for KV cache (~500MB-2GB depending on context size)
    required_memory = model_size_mb * 1.2 + 1024  # 1GB overhead
    
    if gpu_memory_mb >= required_memory:
        # Enough memory for all layers
        return -1
    elif gpu_memory_mb >= model_size_mb * 0.5:
        # Partial offloading: approximate layer count
        # Most models have 32-40 layers for 7-13B models
        ratio = gpu_memory_mb / required_memory
        estimated_layers = int(32 * ratio)
        return max(0, estimated_layers)
    else:
        # Too little GPU memory, use CPU
        return 0


def get_hardware_info() -> HardwareInfo:
    """
    Detect hardware and suggest optimal settings.
    
    Returns:
        HardwareInfo object with detected hardware and suggestions
    """
    gpu_type = detect_gpu()
    gpu_memory_mb = get_gpu_memory() if gpu_type else 0
    cpu_cores, total_ram_mb = get_cpu_info()
    
    # Suggest thread count: use physical cores - 1 to leave room for system
    suggested_threads = max(1, cpu_cores - 1)
    
    # Suggest GPU layers assuming a typical 7B model (~4GB)
    suggested_n_gpu_layers = suggest_n_gpu_layers(4096, gpu_memory_mb)
    
    info = HardwareInfo(
        gpu_type=gpu_type,
        gpu_memory_mb=gpu_memory_mb,
        cpu_cores=cpu_cores,
        total_ram_mb=total_ram_mb,
        suggested_n_gpu_layers=suggested_n_gpu_layers,
        suggested_threads=suggested_threads,
    )
    
    logger.info(f"Hardware detected: {info}")
    return info
