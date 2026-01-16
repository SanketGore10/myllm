"""
llama.cpp wrapper for model loading and inference.
"""

from typing import Iterator, List, Optional, Dict
from pathlib import Path

from llama_cpp import Llama

from app.core.config import get_settings
from app.utils.hardware import get_hardware_info
from app.utils.errors import ModelLoadError, InferenceError
from app.utils.logging import get_logger, PerformanceLogger

logger = get_logger(__name__)


class LlamaCppModel:
    """Thin wrapper around llama-cpp-python."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self._model: Optional[Llama] = None
        self._last_usage: Optional[Dict[str, int]] = None

        if n_threads is None:
            hw = get_hardware_info()
            self.n_threads = hw.suggested_threads

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
                    use_mlock=True,
                    embedding=True,
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
        if not self._model:
            raise InferenceError("Model not loaded")

        from app.engine.sanitizer import OutputSanitizer

        stop_tokens = stop or []
        sanitizer = OutputSanitizer(stop_tokens)

        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = 0

        try:
            with PerformanceLogger(
                logger, f"Generate (max_tokens={max_tokens}, stream={stream})"
            ):
                result = self._model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop_tokens,
                    stream=stream,
                )

                if stream:
                    accumulated_text = ""
                    for chunk in result:
                        if "choices" not in chunk or not chunk["choices"]:
                            continue

                        text = chunk["choices"][0].get("text", "")
                        if not text:
                            continue

                        accumulated_text += text  # Accumulate for accurate count
                        clean = sanitizer.sanitize_token(text)

                        if clean is None:
                            break

                        if clean:
                            yield clean

                    # Count tokens accurately after generation
                    completion_tokens = len(self.tokenize(accumulated_text))

                else:
                    if "choices" in result and result["choices"]:
                        text = result["choices"][0].get("text", "")
                        completion_tokens = len(self.tokenize(text))
                        clean = sanitizer.sanitize(text)
                        yield clean

            self._last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise InferenceError(f"Generation failed: {e}")

    def embed(self, text: str) -> List[float]:
        if not self._model:
            raise InferenceError("Model not loaded")

        try:
            with PerformanceLogger(logger, "Generate embedding"):
                return self._model.embed(text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise InferenceError(f"Embedding generation failed: {e}")

    def tokenize(self, text: str) -> List[int]:
        if not self._model:
            raise InferenceError("Model not loaded")

        try:
            return self._model.tokenize(text.encode("utf-8"))
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise InferenceError(f"Tokenization failed: {e}")

    def detokenize(self, tokens: List[int]) -> str:
        if not self._model:
            raise InferenceError("Model not loaded")

        try:
            return self._model.detokenize(tokens).decode("utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            raise InferenceError(f"Detokenization failed: {e}")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def close(self) -> None:
        if self._model:
            self._model = None
            logger.info(f"Model closed: {self.model_path.name}")

    def __del__(self):
        self.close()


def load_model(
    model_path: Path,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    n_threads: Optional[int] = None,
) -> LlamaCppModel:
    settings = get_settings()

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
