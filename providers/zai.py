"""Z.AI (GLM) model provider implementation."""

import logging
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .openai_compatible import OpenAICompatibleProvider
from .registries.zai import ZAIModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class ZAIModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for Z.AI's GLM models over an OpenAI-compatible endpoint."""

    FRIENDLY_NAME = "Z.AI"

    REGISTRY_CLASS = ZAIModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    PRIMARY_MODEL = "glm-4.6"
    FALLBACK_MODEL = "glm-4.6"

    def __init__(self, api_key: str, **kwargs):
        """Initialize Z.AI provider with API key."""
        kwargs.setdefault("base_url", "https://api.z.ai/api/coding/paas/v4")
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.ZAI

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get Z.AI's preferred model for a given category from allowed models."""
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category in (
            ToolModelCategory.EXTENDED_REASONING,
            ToolModelCategory.FAST_RESPONSE,
            ToolModelCategory.BALANCED,
        ):
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        if self.PRIMARY_MODEL in allowed_models:
            return self.PRIMARY_MODEL
        return allowed_models[0]


# Load registry data at import time
ZAIModelProvider._ensure_registry()
