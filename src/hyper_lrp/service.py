from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from hyper_lrp.backends.lxt import LXTExplainer, LXTExplainerConfig
from hyper_lrp.registry import AdapterRegistry, create_default_registry
from hyper_lrp.types import LRPResult


class LRPService:
    def __init__(
        self,
        registry: AdapterRegistry | None = None,
        explainer: LXTExplainer | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.explainer = explainer or LXTExplainer(LXTExplainerConfig())

    def list_adapters(self) -> list[str]:
        return self.registry.names()

    def explain_image(
        self,
        *,
        adapter_name: str,
        image: Image.Image,
        prompt: str,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LRPResult:
        adapter = self.registry.get(adapter_name)
        return self.explainer.explain(
            adapter=adapter,
            image=image,
            prompt=prompt,
            device=device,
            metadata=metadata,
        )

    def explain_path(
        self,
        *,
        adapter_name: str,
        image_path: str | Path,
        prompt: str,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LRPResult:
        image = Image.open(Path(image_path).expanduser().resolve()).convert("RGB")
        return self.explain_image(
            adapter_name=adapter_name,
            image=image,
            prompt=prompt,
            device=device,
            metadata=metadata,
        )