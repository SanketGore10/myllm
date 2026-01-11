"""
Prompt template builder for different model formats.

Handles model-specific prompt formatting (ChatML, Llama, Alpaca, etc.)
with proper special tokens and structure.
"""

from typing import List

from app.models.schemas import Message
from app.storage.cache import get_template_cache
from app.utils.logging import get_logger

logger = get_logger(__name__)


# Template definitions
TEMPLATES = {
    "chatml": {
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
    },
    "llama3": {
        "bos": "<|begin_of_text|>",
        "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end": "<|eot_id|>",
    },
    "alpaca": {
        "system_start": "### System:\n",
        "system_end": "\n\n",
        "user_start": "### Instruction:\n",
        "user_end": "\n\n",
        "assistant_start": "### Response:\n",
        "assistant_end": "\n\n",
    },
    "vicuna": {
        "system_start": "SYSTEM: ",
        "system_end": "\n",
        "user_start": "USER: ",
        "user_end": "\n",
        "assistant_start": "ASSISTANT: ",
        "assistant_end": "\n",
    },
    "mistral": {
        "bos": "<s>",
        "system_start": "[INST] ",
        "system_end": " [/INST]",
        "user_start": "[INST] ",
        "user_end": " [/INST]",
        "assistant_start": "",
        "assistant_end": "</s>",
    },
}


class PromptBuilder:
    """Builds prompts from messages using model-specific templates."""
    
    def __init__(self, template_name: str = "chatml"):
        """
        Initialize prompt builder.
        
        Args:
            template_name: Name of the template to use
        """
        self.template_name = template_name
        
        if template_name not in TEMPLATES:
            logger.warning(f"Unknown template '{template_name}', using 'chatml'")
            template_name = "chatml"
        
        self.template = TEMPLATES[template_name]
        logger.debug(f"PromptBuilder initialized with template: {template_name}")
    
    def build_prompt(self, messages: List[Message]) -> str:
        """
        Build a prompt string from messages.
        
        Args:
            messages: List of messages
        
        Returns:
            Formatted prompt string
        """
        # Check cache first
        cache = get_template_cache()
        cache_key = f"prompt:{self.template_name}:{hash(str([(m.role, m.content) for m in messages]))}"
        
        cached_prompt = cache.get(cache_key)
        if cached_prompt:
            logger.debug("Using cached prompt")
            return cached_prompt
        
        # Build prompt
        prompt_parts = []
        
        # Add BOS token if present
        if "bos" in self.template:
            prompt_parts.append(self.template["bos"])
        
        # Add messages
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                prompt_parts.append(self.template.get("system_start", ""))
                prompt_parts.append(content)
                prompt_parts.append(self.template.get("system_end", ""))
            
            elif role == "user":
                prompt_parts.append(self.template.get("user_start", ""))
                prompt_parts.append(content)
                prompt_parts.append(self.template.get("user_end", ""))
            
            elif role == "assistant":
                prompt_parts.append(self.template.get("assistant_start", ""))
                prompt_parts.append(content)
                prompt_parts.append(self.template.get("assistant_end", ""))
        
        # Add final assistant prompt to trigger response
        prompt_parts.append(self.template.get("assistant_start", ""))
        
        prompt = "".join(prompt_parts)
        
        # Cache the prompt
        cache.set(cache_key, prompt, ttl=3600)
        
        logger.debug(f"Built prompt: {len(prompt)} chars, {len(messages)} messages")
        
        return prompt
    
    def format_system_message(self, content: str) -> str:
        """
        Format a system message.
        
        Args:
            content: System message content
        
        Returns:
            Formatted system message
        """
        parts = [
            self.template.get("system_start", ""),
            content,
            self.template.get("system_end", ""),
        ]
        return "".join(parts)
    
    def format_user_message(self, content: str) -> str:
        """
        Format a user message.
        
        Args:
            content: User message content
        
        Returns:
            Formatted user message
        """
        parts = [
            self.template.get("user_start", ""),
            content,
            self.template.get("user_end", ""),
        ]
        return "".join(parts)
    
    def format_assistant_message(self, content: str) -> str:
        """
        Format an assistant message.
        
        Args:
            content: Assistant message content
        
        Returns:
            Formatted assistant message
        """
        parts = [
            self.template.get("assistant_start", ""),
            content,
            self.template.get("assistant_end", ""),
        ]
        return "".join(parts)


def detect_template_from_model_name(model_name: str) -> str:
    """
    Detect appropriate template from model name.
    
    Args:
        model_name: Model name
    
    Returns:
        Template name
    """
    model_name_lower = model_name.lower()
    
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return "llama3"
    elif "mistral" in model_name_lower:
        return "mistral"
    elif "alpaca" in model_name_lower:
        return "alpaca"
    elif "vicuna" in model_name_lower:
        return "vicuna"
    else:
        # Default to ChatML
        return "chatml"


def create_prompt_builder(template_name: str) -> PromptBuilder:
    """
    Create a prompt builder for the given template.
    
    Args:
        template_name: Template name
    
    Returns:
        PromptBuilder instance
    """
    return PromptBuilder(template_name)
