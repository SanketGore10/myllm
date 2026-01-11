"""
Tokenizer interface for counting tokens.

Provides token counting without loading the full model (when possible).
Falls back to model tokenizer when necessary.
"""

from typing import Optional, List
from pathlib import Path

from app.models.schemas import Message
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Tokenizer:
    """Tokenizer interface for token counting."""
    
    def __init__(self, model_name: str):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Model name (for logging)
        """
        self.model_name = model_name
        self._model_tokenizer = None  # Lazy-loaded if needed
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        For now, uses a simple approximation. In production, you'd want to:
        1. Load just the tokenizer (not full model) if possible
        2. Cache tokenizer instances
        3. Use model-specific tokenizer
        
        Args:
            text: Input text
        
        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 characters per token on average
        # This is rough but avoids loading full model just for counting
        # For more accuracy, load the actual tokenizer
        approx_tokens = len(text) // 4
        
        # Add overhead for special tokens
        approx_tokens += 3
        
        return max(1, approx_tokens)
    
    def count_messages_tokens(self, messages: List[Message], template: str = "chatml") -> int:
        """
        Count tokens in a list of messages including template overhead.
        
        Args:
            messages: List of messages
            template: Prompt template name
        
        Returns:
            Total token count
        """
        total = 0
        
        # Template-specific overhead per message
        template_overhead = {
            "chatml": 7,     # <|im_start|>role\n{content}<|im_end|>\n
            "llama3": 12,    # <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n{content}<|eot_id|>
            "alpaca": 5,     # ### Role:\n{content}\n\n
            "vicuna": 5,     # ROLE: {content}\n
        }
        
        overhead = template_overhead.get(template, 5)
        
        for message in messages:
            # Count content tokens
            content_tokens = self.count_tokens(message.content)
            
            # Add template overhead
            total += content_tokens + overhead
        
        # Add final assistant prompt overhead
        total += overhead
        
        return total
    
    def estimate_trimmed_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        template: str = "chatml",
    ) -> List[Message]:
        """
        Trim messages to fit within max tokens.
        
        Strategy:
        1. Always keep system message (if present)
        2. Always keep last user message
        3. Remove oldest messages until under limit
        4. Preserve at least last 2 turns if possible
        
        Args:
            messages: List of messages
            max_tokens: Maximum token budget
            template: Prompt template name
        
        Returns:
            Trimmed list of messages
        """
        if not messages:
            return []
        
        # Separate system messages from conversation
        system_messages = [m for m in messages if m.role == "system"]
        conversation_messages = [m for m in messages if m.role != "system"]
        
        # Count system message tokens
        system_tokens = sum(self.count_tokens(m.content) + 7 for m in system_messages)
        
        # Reserve tokens for system messages
        available_tokens = max_tokens - system_tokens
        
        if available_tokens <= 0:
            logger.warning("System messages exceed token budget")
            return system_messages[:1] if system_messages else []
        
        # Ensure we keep at least the last user message
        if not conversation_messages:
            return system_messages
        
        last_message = conversation_messages[-1]
        last_message_tokens = self.count_tokens(last_message.content) + 7
        
        if last_message_tokens > available_tokens:
            logger.warning("Last message exceeds available tokens")
            # Truncate last message content if needed
            # For now, just include system + truncated last message
            return system_messages + [last_message]
        
        # Build from end, keeping as many recent messages as possible
        trimmed_conversation = []
        current_tokens = last_message_tokens
        
        # Start from second-to-last and work backwards
        for message in reversed(conversation_messages[:-1]):
            message_tokens = self.count_tokens(message.content) + 7
            
            if current_tokens + message_tokens <= available_tokens:
                trimmed_conversation.insert(0, message)
                current_tokens += message_tokens
            else:
                # Can't fit any more messages
                break
        
        # Add the last message
        trimmed_conversation.append(last_message)
        
        # Combine system + trimmed conversation
        result = system_messages + trimmed_conversation
        
        logger.debug(
            f"Trimmed messages: {len(messages)} -> {len(result)} "
            f"(est. tokens: {self.count_messages_tokens(result, template)}/{max_tokens})"
        )
        
        return result


def create_tokenizer(model_name: str) -> Tokenizer:
    """
    Create a tokenizer for the given model.
    
    Args:
        model_name: Model name
    
    Returns:
        Tokenizer instance
    """
    return Tokenizer(model_name)
