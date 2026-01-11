"""
Model family template definitions - EXPLICIT, NO GUESSING.

Based on actual model training formats. Each template includes:
- Exact prompt format
- Stop tokens (enforced during generation)
- BOS/EOS tokens
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Prompt template with format strings and stop tokens."""
    
    name: str
    system_format: str
    user_format: str
    assistant_format: str
    bos_token: str
    eos_token: str
    stop_tokens: List[str]
    
    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build prompt from messages using exact template format."""
        parts = [self.bos_token] if self.bos_token else []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                parts.append(self.system_format.format(content=content))
            elif role == "user":
                parts.append(self.user_format.format(content=content))
            elif role == "assistant":
                parts.append(self.assistant_format.format(content=content))
        
        # Add assistant prefix WITHOUT content for generation
        if messages and messages[-1]["role"] != "assistant":
            parts.append(self.assistant_format.split("{content}")[0])
        
        return "".join(parts)


# TEMPLATE REGISTRY - IMMUTABLE, EXPLICIT
TEMPLATES: Dict[str, PromptTemplate] = {
    # LLaMA family (TinyLlama, LLaMA 1, LLaMA 2)
    "llama": PromptTemplate(
        name="llama",
        system_format="<<SYS>>\n{content}\n<</SYS>>\n\n",
        user_format="[INST] {content} [/INST]",
        assistant_format="{content}</s>",
        bos_token="<s>",
        eos_token="</s>",
        stop_tokens=["</s>", "[INST]"],
    ),
    
    # LLaMA 3
    "llama3": PromptTemplate(
        name="llama3",
        system_format="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        user_format="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        assistant_format="<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        stop_tokens=["<|eot_id|>"],
    ),
    
    # Mistral
    "mistral": PromptTemplate(
        name="mistral",
        system_format="<<SYS>>\n{content}\n<</SYS>>\n\n",
        user_format="[INST] {content} [/INST]",
        assistant_format="{content}</s>",
        bos_token="<s>",
        eos_token="</s>",
        stop_tokens=["</s>"],
    ),
    
    # Phi
    "phi": PromptTemplate(
        name="phi",
        system_format="### System:\n{content}\n\n",
        user_format="### Instruction:\n{content}\n\n",
        assistant_format="### Response:\n{content}\n\n",
        bos_token="",
        eos_token="",
        stop_tokens=["###"],
    ),
    
    # Qwen
    "qwen": PromptTemplate(
        name="qwen",
        system_format="<|im_start|>system\n{content}<|im_end|>\n",
        user_format="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_format="<|im_start|>assistant\n{content}<|im_end|>\n",
        bos_token="",
        eos_token="<|im_end|>",
        stop_tokens=["<|im_end|>"],
    ),
}


def get_template(family: str) -> PromptTemplate:
    """
    Get template for model family.
    
    CRITICAL: This function MUST NOT guess or provide defaults.
    If template is not found, system must fail loudly.
    """
    if family not in TEMPLATES:
        raise ValueError(
            f"No template found for family '{family}'. "
            f"Available families: {list(TEMPLATES.keys())}. "
            f"Templates MUST be explicit - guessing is not allowed."
        )
    
    return TEMPLATES[family]
