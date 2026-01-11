"""Unit tests for app/core/templates.py - Template registry and formatting."""

import pytest

from app.core.templates import (
    TEMPLATES,
    PromptTemplate,
    get_template,
)


class TestTemplateRegistry:
    """Test template registry."""
    
    def test_all_families_present(self):
        """Test that all expected model families have templates."""
        expected_families = ["llama", "llama3", "mistral", "phi", "qwen"]
        
        for family in expected_families:
            assert family in TEMPLATES, f"Missing template for family: {family}"
    
    def test_templates_have_required_fields(self):
        """Test that all templates have required fields."""
        for family, template in TEMPLATES.items():
            assert isinstance(template, PromptTemplate), f"{family} is not a PromptTemplate"
            assert template.name == family
            assert isinstance(template.stop_tokens, list)
            assert len(template.stop_tokens) > 0, f"{family} has no stop tokens"
    
    def test_get_template_success(self):
        """Test get_template returns correct template."""
        template = get_template("llama")
        
        assert template.name == "llama"
        assert "</s>" in template.stop_tokens
    
    def test_get_template_invalid_family_raises(self):
        """Test get_template raises ValueError for invalid family."""
        with pytest.raises(ValueError, match="No template found"):
            get_template("invalid_family")


class TestLLaMATemplate:
    """Test LLaMA template formatting."""
    
    def test_single_user_message(self):
        """Test formatting single user message."""
        template = get_template("llama")
        messages = [{"role": "user", "content": "Hello"}]
        
        prompt = template.build_prompt(messages)
        
        # Should have <s> at start and [INST] format
        assert prompt.startswith("<s>[INST]")
        assert "Hello" in prompt
        assert "[/INST]" in prompt
    
    def test_no_duplicate_bos(self):
        """Test that <s> token is not duplicated."""
        template = get_template("llama")
        messages = [{"role": "user", "content": "Test"}]
        
        prompt = template.build_prompt(messages)
        
        # Should only have one <s> at the beginning
        assert prompt.count("<s>") == 1
        assert prompt.startswith("<s>")
    
    def test_conversation_format(self):
        """Test multi-turn conversation formatting."""
        template = get_template("llama")
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        
        prompt = template.build_prompt(messages)
        
        # Should have both messages
        assert "Hi" in prompt
        assert "Hello!" in prompt
        assert "How are you?" in prompt
        
        # Should end with [/INST] to prompt assistant
        assert "[/INST]" in prompt.split("How are you?")[-1]
    
    def test_stop_tokens(self):
        """Test LLaMA stop tokens."""
        template = get_template("llama")
        
        assert "</s>" in template.stop_tokens
        assert "[INST]" in template.stop_tokens


class TestPhiTemplate:
    """Test Phi template formatting."""
    
    def test_phi_format(self):
        """Test Phi uses Alpaca-style format."""
        template = get_template("phi")
        messages = [{"role": "user", "content": "Test question"}]
        
        prompt = template.build_prompt(messages)
        
        assert "### Instruction:" in prompt
        assert "Test question" in prompt
        assert "### Response:" in prompt
    
    def test_phi_stop_tokens(self):
        """Test Phi stop tokens."""
        template = get_template("phi")
        
        assert "###" in template.stop_tokens


class TestQwenTemplate:
    """Test Qwen template formatting."""
    
    def test_qwen_format(self):
        """Test Qwen uses ChatML format."""
        template = get_template("qwen")
        messages = [{"role": "user", "content": "Hello"}]
        
        prompt = template.build_prompt(messages)
        
        assert "<|im_start|>user" in prompt
        assert "Hello" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant" in prompt
    
    def test_qwen_stop_tokens(self):
        """Test Qwen stop tokens."""
        template = get_template("qwen")
        
        assert "<|im_end|>" in template.stop_tokens


class TestLLaMA3Template:
    """Test LLaMA 3 template formatting."""
    
    def test_llama3_format(self):
        """Test LLaMA 3 uses header_id format."""
        template = get_template("llama3")
        messages = [{"role": "user", "content": "Test"}]
        
        prompt = template.build_prompt(messages)
        
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "Test" in prompt
        assert "<|eot_id|>" in prompt
    
    def test_llama3_stop_tokens(self):
        """Test LLaMA 3 stop tokens."""
        template = get_template("llama3")
        
        assert "<|eot_id|>" in template.stop_tokens
