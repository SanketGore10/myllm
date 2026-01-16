"""
Prompt builder for MyLLM.

RESPONSIBILITY RULES (NON-NEGOTIABLE):
- This module builds TEXT ONLY prompts.
- It MUST NOT emit BOS (<s>) or EOS (</s>) tokens.
- It MUST NOT emit stop tokens.
- It MUST NOT guess templates.
- Tokenization concerns belong to llama.cpp runtime.
"""

from typing import List, Dict, Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """
    Builds prompts using an explicit, model-family-specific template.

    Accepts ONLY normalized messages:
    {"role": str, "content": str}

    Message objects are tolerated but NOT required.
    """

    def __init__(self, family: str):
        from app.core.templates import get_template

        self.family = family
        self.template = get_template(family)  # MUST exist

        logger.debug("PromptBuilder initialized", extra={"family": family})

    def build_prompt(self, messages: List[Any]) -> str:
        """
        Build a prompt string from structured messages.

        Args:
            messages: List of dicts or Message-like objects

        Returns:
            Prompt string (TEXT ONLY, no BOS/EOS)

        Raises:
            RuntimeError: If BOS/EOS tokens leak into the prompt
        """
        if not messages:
            raise ValueError("Cannot build prompt from empty message list")

        normalized: List[Dict[str, str]] = []

        for m in messages:
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
            else:
                # Fallback for Message objects (defensive)
                role = getattr(m, "role", None)
                content = getattr(m, "content", None)

            if not role or content is None:
                raise ValueError(f"Invalid message format: {m}")

            normalized.append({"role": role, "content": content})

        prompt = self.template.build_prompt(normalized)

        # 🚨 HARD GUARDS — MUST NEVER TRIGGER
        stripped = prompt.lstrip()

        if "<s>" in stripped:
            raise RuntimeError(
                "Invalid prompt: BOS token '<s>' leaked into prompt."
            )

        if "</s>" in stripped:
            raise RuntimeError(
                "Invalid prompt: EOS token '</s>' leaked into prompt."
            )

        if "<|begin_of_text|>" in stripped:
            raise RuntimeError(
                "Invalid prompt: LLaMA3 BOS token leaked into prompt."
            )

        logger.debug(
            "Prompt built successfully",
            extra={"family": self.family, "length": len(prompt)},
        )

        return prompt

    def get_stop_tokens(self) -> List[str]:
        """
        Return stop tokens associated with this template.
        Stop tokens are enforced by runtime, NOT included in prompt.
        """
        return list(self.template.stop_tokens)


def create_prompt_builder(family: str) -> PromptBuilder:
    """Factory for PromptBuilder."""
    return PromptBuilder(family)



# Commented out old implementation for reference
# Prompt template builder for different model formats.

# Handles model-specific prompt formatting (ChatML, Llama, Alpaca, etc.)
# with proper special tokens and structure.
# """

# from typing import List

# from app.models.schemas import Message
# from app.storage.cache import get_template_cache
# from app.utils.logging import get_logger

# logger = get_logger(__name__)


# # Template definitions
# TEMPLATES = {
#     "chatml": {
#         "system_start": "<|im_start|>system\n",
#         "system_end": "<|im_end|>\n",
#         "user_start": "<|im_start|>user\n",
#         "user_end": "<|im_end|>\n",
#         "assistant_start": "<|im_start|>assistant\n",
#         "assistant_end": "<|im_end|>\n",
#     },
#     "llama3": {
#         "bos": "<|begin_of_text|>",
#         "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
#         "system_end": "<|eot_id|>",
#         "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
#         "user_end": "<|eot_id|>",
#         "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
#         "assistant_end": "<|eot_id|>",
#     },
#     "alpaca": {
#         "system_start": "### System:\n",
#         "system_end": "\n\n",
#         "user_start": "### Instruction:\n",
#         "user_end": "\n\n",
#         "assistant_start": "### Response:\n",
#         "assistant_end": "\n\n",
#     },
#     "vicuna": {
#         "system_start": "SYSTEM: ",
#         "system_end": "\n",
#         "user_start": "USER: ",
#         "user_end": "\n",
#         "assistant_start": "ASSISTANT: ",
#         "assistant_end": "\n",
#     },
#     "mistral": {
#         "bos": "<s>",
#         "system_start": "[INST] ",
#         "system_end": " [/INST]",
#         "user_start": "[INST] ",
#         "user_end": " [/INST]",
#         "assistant_start": "",
#         "assistant_end": "</s>",
#     },
# }


# class PromptBuilder:
#     """
#     Builds prompts from messages using model-specific templates.
    
#     CRITICAL: Uses explicit template registry from app.core.templates.
#     NO GUESSING ALLOWED. Template must come from model config.
#     """
    
#     def __init__(self, family: str):
#         """
#         Initialize prompt builder with model family.
        
#         Args:
#             family: Model family (llama, phi, qwen, etc.)
        
#         Raises:
#             ValueError: If family has no defined template
#         """
#         from app.core.templates import get_template
        
#         # Get template from explicit registry (fails if not found)
#         self.template_obj = get_template(family)
#         self.family = family
        
#         logger.debug(f"PromptBuilder initialized with family: {family}")
    
#     def build_prompt(self, messages: List[Message]) -> str:
#         """
#         Build a prompt string from messages using explicit template.
        
#         Args:
#             messages: List of messages
        
#         Returns:
#             Formatted prompt string
#         """
#         # Use template from app.core.templates
#         return self.template_obj.build_prompt(
#             [{"role": m.role, "content": m.content} for m in messages]
#         )
    
#     def get_stop_tokens(self) -> List[str]:
#         """
#         Get stop tokens for this template.
        
#         Returns:
#             List of stop tokens
#         """
#         return self.template_obj.stop_tokens
    
#     def format_system_message(self, content: str) -> str:
#         """
#         Format a system message.
        
#         Args:
#             content: System message content
        
#         Returns:
#             Formatted system message
#         """
#         parts = [
#             self.template.get("system_start", ""),
#             content,
#             self.template.get("system_end", ""),
#         ]
#         return "".join(parts)
    
#     def format_user_message(self, content: str) -> str:
#         """
#         Format a user message.
        
#         Args:
#             content: User message content
        
#         Returns:
#             Formatted user message
#         """
#         parts = [
#             self.template.get("user_start", ""),
#             content,
#             self.template.get("user_end", ""),
#         ]
#         return "".join(parts)
    
#     def format_assistant_message(self, content: str) -> str:
#         """
#         Format an assistant message.
        
#         Args:
#             content: Assistant message content
        
#         Returns:
#             Formatted assistant message
#         """
#         parts = [
#             self.template.get("assistant_start", ""),
#             content,
#             self.template.get("assistant_end", ""),
#         ]
#         return "".join(parts)



# def create_prompt_builder(family: str) -> PromptBuilder:
#     """
#     Create a prompt builder for the given model family.
    
#     Args:
#         family: Model family (llama, phi, qwen, etc.)
    
#     Returns:
#         PromptBuilder instance
    
#     Raises:
#         ValueError: If family has no defined template
#     """
#     return PromptBuilder(family)

