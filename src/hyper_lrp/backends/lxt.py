from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hyper_lrp.heatmap import normalize_signed_heatmap, overlay_signed_heatmap, pool_to_patches
from hyper_lrp.types import LRPResult


@dataclass(slots=True)
class LXTExplainerConfig:
    gamma_conv: float = 0.25
    gamma_linear: float = 0.05
    overlay_alpha: float = 0.52


class LXTExplainer:
    _zennit_patched = False

    def __init__(self, config: LXTExplainerConfig) -> None:
        self.config = config

    def explain(
        self,
        *,
        adapter: Any,
        image: Any,
        prompt: str,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LRPResult:
        lrp_mode = getattr(adapter, "lrp_mode", "zennit-gamma-fallback")

        if lrp_mode == "attnlrp-efficient":
            return self._explain_attnlrp_efficient(
                adapter=adapter,
                image=image,
                prompt=prompt,
                device=device,
                metadata=metadata,
            )

        return self._explain_zennit_fallback(
            adapter=adapter,
            image=image,
            prompt=prompt,
            device=device,
            metadata=metadata,
        )

    def _explain_attnlrp_efficient(
        self,
        *,
        adapter: Any,
        image: Any,
        prompt: str,
        device: str | None,
        metadata: dict[str, Any] | None,
    ) -> LRPResult:
        import torch
        from zennit.composites import LayerMapComposite
        import zennit.rules as z_rules

        # Zennit hooks must be made compatible with the efficient Gradient×Input
        # framework. This is a global one-time patch.
        if not self.__class__._zennit_patched:
            from lxt.efficient import monkey_patch_zennit

            monkey_patch_zennit(verbose=False)
            self.__class__._zennit_patched = True

        resolved_device = adapter.resolve_device(device)
        adapter.ensure_patched()
        prepared = adapter.prepare(prompt=prompt, device=resolved_device)

        # Gamma-rule composite for Conv2d (patch embedding) and Linear layers.
        # This provides the ViT-specific denoising that the AttnLRP paper
        # recommends alongside the efficient attention/MLP/LayerNorm patches.
        composite = LayerMapComposite(
            [
                (torch.nn.Conv2d, z_rules.Gamma(self.config.gamma_conv)),
                (torch.nn.Linear, z_rules.Gamma(self.config.gamma_linear)),
            ]
        )

        pixel_values, display_image = prepared.prepare_image(image)
        pixel_values = pixel_values.to(resolved_device).requires_grad_(True)
        prepared.scorer.zero_grad(set_to_none=True)

        composite.register(prepared.scorer)
        try:
            scores = prepared.scorer(pixel_values)
            score = scores.reshape(-1).sum()
            score.backward()
        finally:
            composite.remove()

        if pixel_values.grad is None:
            raise RuntimeError("Input gradient is missing after backward pass")

        raw_heatmap = (pixel_values.grad * pixel_values).sum(dim=1)[0].detach().cpu().float().numpy()

        # ViT models produce patch-level features; average within patches
        # to remove sub-patch speckle from the Conv2d backward pass.
        patch_size = getattr(adapter, "patch_size", 0)
        if patch_size > 1:
            raw_heatmap = pool_to_patches(raw_heatmap, patch_size)

        normalized = normalize_signed_heatmap(np.asarray(raw_heatmap, dtype=np.float32))
        overlay = overlay_signed_heatmap(
            display_image,
            normalized,
            alpha=self.config.overlay_alpha,
        )

        merged_metadata = dict(prepared.metadata)
        merged_metadata["lrp_mode"] = "attnlrp-efficient"
        merged_metadata["gamma_conv"] = self.config.gamma_conv
        merged_metadata["gamma_linear"] = self.config.gamma_linear
        if metadata:
            merged_metadata.update(metadata)

        return LRPResult(
            adapter_name=adapter.name,
            prompt=prompt,
            score=float(score.detach().cpu().item()),
            heatmap=normalized,
            display_image=display_image,
            overlay_image=overlay,
            metadata=merged_metadata,
        )

    def _explain_zennit_fallback(
        self,
        *,
        adapter: Any,
        image: Any,
        prompt: str,
        device: str | None,
        metadata: dict[str, Any] | None,
    ) -> LRPResult:
        import torch
        from zennit.composites import LayerMapComposite
        import zennit.rules as z_rules

        resolved_device = adapter.resolve_device(device)
        adapter.ensure_patched()
        prepared = adapter.prepare(prompt=prompt, device=resolved_device)

        if not self.__class__._zennit_patched:
            from lxt.efficient import monkey_patch_zennit

            monkey_patch_zennit(verbose=False)
            self.__class__._zennit_patched = True

        composite = LayerMapComposite(
            [
                (torch.nn.Conv2d, z_rules.Gamma(self.config.gamma_conv)),
                (torch.nn.Linear, z_rules.Gamma(self.config.gamma_linear)),
            ]
        )

        pixel_values, display_image = prepared.prepare_image(image)
        pixel_values = pixel_values.to(resolved_device).requires_grad_(True)
        prepared.scorer.zero_grad(set_to_none=True)

        composite.register(prepared.scorer)
        try:
            scores = prepared.scorer(pixel_values)
            score = scores.reshape(-1).sum()
            score.backward()
        finally:
            composite.remove()

        if pixel_values.grad is None:
            raise RuntimeError("Input gradient is missing after backward pass")

        heatmap = (pixel_values.grad * pixel_values).sum(dim=1)[0].detach().cpu().float().numpy()
        normalized = normalize_signed_heatmap(np.asarray(heatmap, dtype=np.float32))
        overlay = overlay_signed_heatmap(
            display_image,
            normalized,
            alpha=self.config.overlay_alpha,
        )

        merged_metadata = dict(prepared.metadata)
        merged_metadata.setdefault("lrp_mode", "zennit-gamma-fallback")
        if metadata:
            merged_metadata.update(metadata)

        return LRPResult(
            adapter_name=adapter.name,
            prompt=prompt,
            score=float(score.detach().cpu().item()),
            heatmap=normalized,
            display_image=display_image,
            overlay_image=overlay,
            metadata=merged_metadata,
        )