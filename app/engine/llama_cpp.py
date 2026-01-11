"""
llama.cpp wrapper for model loading and inference.

Thin wrapper around llama-cpp-python that handles model loading,
tokenization, generation, and embeddings.
"""

from typing import Iterator, List, Optional, Dict, Any
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar

from app.core.config import get_settings
from app.utils.hardware import get_hardware_info
from app.utils.errors import ModelLoadError, InferenceError
from app.utils.logging import get_logger, PerformanceLogger

logger = get_logger(__name__)


class LlamaCppModel:
    """Wrapper around llama-cpp-python Llama model."""
    
    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize llama.cpp model.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_threads: Number of threads (None = auto-detect)
            verbose: Enable verbose logging from llama.cpp
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self._model: Optional[Llama] = None
        
        # Auto-detect threads if not specified
        if n_threads is None:
            hw_info = get_hardware_info()
            self.n_threads = hw_info.suggested_threads
        
        logger.info(
            f"Initializing llama.cpp model: {model_path.name} "
            f"(ctx={n_ctx}, gpu_layers={n_gpu_layers}, threads={self.n_threads})"
        )
        
        try:
            with PerformanceLogger(logger, f"Load model {model_path.name}"):
                self._model = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=self.n_threads,
                    verbose=verbose,
                    use_mlock=True,  # Keep model in RAM
                    embedding=True,  # Enable embeddings
                )
            
            logger.info(f"Model loaded successfully: {model_path.name}")
        
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise ModelLoadError(model_path.name, str(e))
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        """
        Generate text from prompt with streaming support.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: Stop sequences
            stream: Enable streaming (yields tokens)
        
        Yields:
            Generated tokens (if stream=True)
        
        Returns:
            Complete generated text (if stream=False)
        
        Raises:
            InferenceError: If generation fails
        """
        if not self._model:
            raise InferenceError("Model not loaded")
        
        try:
            with PerformanceLogger(logger, f"Generate (max_tokens={max_tokens}, stream={stream})"):
                # Call llama.cpp completion
                result = self._model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    stream=stream,
                )
                
                if stream:
                    # Streaming mode: yield tokens
                    for chunk in result:
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            text = chunk["choices"][0].get("text", "")
                            if text:
                                yield text
                else:
                    # Non-streaming: return full text
                    if "choices" in result and len(result["choices"]) > 0:
                        text = result["choices"][0].get("text", "")
                        yield text
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise InferenceError(f"Generation failed: {e}")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector as list of floats
        
        Raises:
            InferenceError: If embedding fails
        """
        if not self._model:
            raise InferenceError("Model not loaded")
        
        try:
            with PerformanceLogger(logger, "Generate embedding"):
                embedding = self._model.embed(text)
                return embedding
        
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise InferenceError(f"Embedding generation failed: {e}")
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text to token IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        if not self._model:
            raise InferenceError("Model not loaded")
        
        try:
            tokens = self._model.tokenize(text.encode("utf-8"))
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise InferenceError(f"Tokenization failed: {e}")
    
    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenize token IDs to text.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            Decoded text
        """
        if not self._model:
            raise InferenceError("Model not loaded")
        
        try:
            text = self._model.detokenize(tokens).decode("utf-8", errors="ignore")
            return text
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            raise InferenceError(f"Detokenization failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Token count
        """
        tokens = self.tokenize(text)
        return len(tokens)
    
    def close(self) -> None:
        """Close model and free resources."""
        if self._model:
            # llama-cpp-python doesn't have explicit close, rely on garbage collection
            self._model = None
            logger.info(f"Model closed: {self.model_path.name}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def load_model(
    model_path: Path,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    n_threads: Optional[int] = None,
) -> LlamaCppModel:
    """
    Load a GGUF model with llama.cpp.
    
    Args:
        model_path: Path to GGUF file
        n_ctx: Context size (None = use default from settings)
        n_gpu_layers: GPU layers (None = use default from settings)
        n_threads: Thread count (None = auto-detect)
    
    Returns:
        Loaded LlamaCppModel instance
    
    Raises:
        ModelLoadError: If model fails to load
    """
    settings = get_settings()
    
    # Use defaults from settings if not specified
    if n_ctx is None:
        n_ctx = settings.default_context_size
    
    if n_gpu_layers is None:
        n_gpu_layers = settings.default_n_gpu_layers
    
    return LlamaCppModel(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
    )
