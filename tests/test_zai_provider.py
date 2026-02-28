"""Tests for Z.AI provider implementation."""

import os
from unittest.mock import patch

import pytest

from providers.shared import ProviderType
from providers.zai import ZAIModelProvider


class TestZAIProvider:
    """Test Z.AI provider functionality."""

    def setup_method(self):
        """Clear restriction service cache before each test."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clear restriction service cache after each test."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def test_initialization(self):
        """Test provider initialization."""
        provider = ZAIModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.ZAI
        assert provider.base_url == "https://api.z.ai/api/coding/paas/v4"

    def test_model_validation(self):
        """Test model validation and aliases."""
        provider = ZAIModelProvider("test-key")

        assert provider.validate_model_name("glm-4.6") is True
        assert provider.validate_model_name("glm") is True
        assert provider.validate_model_name("glm-4") is True
        assert provider.validate_model_name("glm4.6") is True

        assert provider.validate_model_name("invalid-model") is False
        assert provider.validate_model_name("grok-4") is False

    def test_resolve_model_name(self):
        """Test alias resolution."""
        provider = ZAIModelProvider("test-key")

        assert provider._resolve_model_name("glm") == "glm-4.6"
        assert provider._resolve_model_name("glm-4") == "glm-4.6"
        assert provider._resolve_model_name("glm4.6") == "glm-4.6"
        assert provider._resolve_model_name("glm-4.6") == "glm-4.6"

    def test_get_capabilities(self):
        """Test capabilities for GLM-4.6."""
        provider = ZAIModelProvider("test-key")

        capabilities = provider.get_capabilities("glm")
        assert capabilities.model_name == "glm-4.6"
        assert capabilities.provider == ProviderType.ZAI
        assert capabilities.context_window == 128_000
        assert capabilities.max_output_tokens == 8_192
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_images is True
        assert capabilities.supports_temperature is True

    def test_invalid_model_capabilities(self):
        """Unsupported models should raise."""
        provider = ZAIModelProvider("test-key")

        with pytest.raises(ValueError, match="Unsupported model 'invalid-model' for provider zai"):
            provider.get_capabilities("invalid-model")

    def test_get_preferred_model_respects_allowed_models(self):
        """Preferred model selection must respect allow-list input."""
        from tools.models import ToolModelCategory

        provider = ZAIModelProvider("test-key")

        assert provider.get_preferred_model(ToolModelCategory.BALANCED, ["glm-4.6"]) == "glm-4.6"
        assert provider.get_preferred_model(ToolModelCategory.BALANCED, ["glm"]) == "glm"

    @patch.dict(os.environ, {"ZAI_ALLOWED_MODELS": "glm-4.6"})
    def test_model_restrictions(self):
        """Test Z.AI allow-list restrictions."""
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = ZAIModelProvider("test-key")

        assert provider.validate_model_name("glm-4.6") is True
        assert provider.validate_model_name("glm") is True
        assert provider.validate_model_name("glm4.6") is True

    @patch.dict(os.environ, {"ZAI_ALLOWED_MODELS": "glm"})
    def test_alias_allowlist_resolution(self):
        """Alias allow-list entries should resolve to canonical model."""
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = ZAIModelProvider("test-key")
        assert provider.validate_model_name("glm-4.6") is True
        assert provider.validate_model_name("glm") is True
