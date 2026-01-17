"""
Context managers for clean UX.

Provides utilities to suppress logs and stderr for production-grade output.
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional


@contextmanager
def suppress_logs(level: int = logging.ERROR):
    """
    Temporarily suppress logs below a certain level.
    
    Usage:
        with suppress_logs():
            # Code here won't show INFO/WARNING logs
            do_something()
    
    Args:
        level: Minimum log level to show (default: ERROR)
    """
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(old_level)


@contextmanager
def suppress_stderr():
    """
    Suppress stderr output (for llama.cpp noise).
    
    Usage:
        with suppress_stderr():
            # llama.cpp warnings won't show
            model.generate(...)
    
    CRITICAL: This prevents llama.cpp from polluting user chat with:
    - "init: embeddings required but some input tokens..."
    - "llama_model_loader: loaded meta data..."
    - etc.
    
    """
    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)  # Duplicate stderr fd
    
    try:
        # Redirect stderr to devnull
        os.dup2(devnull_fd, 2)
        sys.stderr.flush()
        yield
    finally:
        # Restore stderr
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


@contextmanager
def quiet_mode():
    """
    Combined: suppress both logs and stderr.
    
    Perfect for interactive chat where user expects clean output.
    
    Usage:
        with quiet_mode():
            response = await runtime.chat(...)
    """
    with suppress_logs(), suppress_stderr():
        yield
