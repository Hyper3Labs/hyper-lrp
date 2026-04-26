from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from hyper_lrp.adapters.base import PreparedAdapter
from hyper_lrp.adapters.hf_clip import HuggingFaceCLIPAdapter
from hyper_lrp.backends.lxt import LXTExplainer, LXTExplainerConfig


class _DummyScorer(nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.mean(dim=(1, 2, 3))


@dataclass(slots=True)
class _DummyAdapter:
    name: str = "dummy"
    lrp_mode: str = "attnlrp-efficient"

    @staticmethod
    def resolve_device(device: str | None) -> str:
        return device or "cpu"

    def ensure_patched(self) -> None:
        return

    def prepare(self, *, prompt: str, device: str) -> PreparedAdapter:
        scorer = _DummyScorer().to(device)

        def prepare_image(image: Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
            return tensor, image.convert("RGB")

        return PreparedAdapter(
            scorer=scorer,
            prepare_image=prepare_image,
            metadata={"backend_mode": "attnlrp-efficient-dummy", "prompt": prompt},
        )


def test_hf_clip_adapter_applies_attnlrp_patch_map() -> None:
    from transformers.models.clip import modeling_clip

    adapter = HuggingFaceCLIPAdapter()
    adapter.ensure_patched()

    assert adapter.lrp_mode == "attnlrp-efficient"
    assert modeling_clip.CLIPMLP.forward.__module__ == "hyper_lrp.adapters.hf_clip"
    assert modeling_clip.eager_attention_forward.__module__ == "lxt.efficient.patches"


def test_explainer_uses_attnlrp_branch_for_attnlrp_adapter() -> None:
    explainer = LXTExplainer(LXTExplainerConfig())
    image = Image.new("RGB", (16, 16), color=(140, 90, 60))

    result = explainer.explain(
        adapter=_DummyAdapter(),
        image=image,
        prompt="test prompt",
        device="cpu",
    )

    assert result.metadata["lrp_mode"] == "attnlrp-efficient"
    assert result.metadata["backend_mode"] == "attnlrp-efficient-dummy"
    assert result.metadata["gamma_conv"] == 0.25
    assert result.metadata["gamma_linear"] == 0.05


def test_patch_pooling_applied_when_adapter_has_patch_size() -> None:
    explainer = LXTExplainer(LXTExplainerConfig())
    image = Image.new("RGB", (32, 32), color=(140, 90, 60))

    @dataclass(slots=True)
    class _PatchAdapter(_DummyAdapter):
        patch_size: int = 8

        def prepare(self, *, prompt: str, device: str) -> PreparedAdapter:
            # Use a Conv2d + Linear scorer so Zennit can attach Gamma hooks
            scorer = nn.Sequential(
                nn.Conv2d(3, 1, 1),
                nn.Flatten(),
                nn.Linear(32 * 32, 1),
            ).to(device)

            def prepare_image(image: Image.Image):
                arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
                return tensor, image.convert("RGB")

            return PreparedAdapter(
                scorer=scorer,
                prepare_image=prepare_image,
                metadata={"backend_mode": "attnlrp-efficient-dummy", "prompt": prompt},
            )

    result = explainer.explain(
        adapter=_PatchAdapter(),
        image=image,
        prompt="test prompt",
        device="cpu",
    )

    # With patch_size=8 on a 32×32 image, the heatmap should have
    # uniform values within each 8×8 block (from pool_to_patches)
    h = result.heatmap
    block = h[:8, :8]
    assert np.allclose(block, block[0, 0], atol=1e-7), "patch pooling should make each 8×8 block uniform"
    assert np.isfinite(result.heatmap).all()
    assert float(np.abs(result.heatmap).sum()) > 0.0
