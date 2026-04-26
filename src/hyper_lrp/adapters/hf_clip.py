from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from PIL import Image
import torch.nn as nn

from hyper_lrp.adapters.base import BaseAdapter, PreparedAdapter
from hyper_lrp.heatmap import tensor_image_to_pil


def _clip_mlp_attnlrp_forward(self: Any, hidden_states: Any) -> Any:
    from lxt.efficient.rules import identity_rule_implicit

    hidden_states = self.fc1(hidden_states)
    hidden_states = identity_rule_implicit(self.activation_fn, hidden_states)
    hidden_states = self.fc2(hidden_states)
    return hidden_states


def _patch_hf_clip_for_attnlrp() -> None:
    from lxt.efficient import monkey_patch
    from lxt.efficient.patches import layer_norm_forward, patch_attention, patch_method
    from transformers.models.clip import modeling_clip

    patch_map = {
        modeling_clip.CLIPMLP: partial(patch_method, _clip_mlp_attnlrp_forward),
        modeling_clip.nn.LayerNorm: partial(patch_method, layer_norm_forward),
        modeling_clip: patch_attention,
    }
    monkey_patch(modeling_clip, patch_map=patch_map, verbose=False)


class _HFCLIPScorer(nn.Module):
    def __init__(self, model: Any, text_features: Any):
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.model.zero_grad(set_to_none=set_to_none)

    def forward(self, pixel_values: Any) -> Any:
        from lxt.efficient.rules import stop_gradient

        image_features = self.model.get_image_features(pixel_values=pixel_values)
        # Apply identity-rule-style normalization by stopping relevance through the norm term.
        image_norm = image_features.norm(dim=-1, keepdim=True)
        image_features = image_features / stop_gradient(image_norm + 1e-12)
        return (image_features * self.text_features).sum(dim=-1)


@dataclass(slots=True)
class HuggingFaceCLIPAdapter(BaseAdapter):
    model_id: str = "openai/clip-vit-base-patch32"

    def __post_init__(self) -> None:
        self.name = "hf-clip"
        self.lrp_mode = "attnlrp-efficient"
        self.patch_size = 32  # ViT-B/32 patch size for sub-patch pooling
        self._patched = False
        self._cache: dict[str, tuple[Any, Any]] = {}

    def ensure_patched(self) -> None:
        if self._patched:
            return
        _patch_hf_clip_for_attnlrp()
        self._patched = True

    def _load(self, device: str) -> tuple[Any, Any]:
        cached = self._cache.get(device)
        if cached is not None:
            return cached

        import torch
        from transformers import AutoProcessor, CLIPModel

        self.ensure_patched()
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = CLIPModel.from_pretrained(self.model_id, attn_implementation="eager").to(device).eval()
        model.config._attn_implementation = "eager"
        model.text_model.config._attn_implementation = "eager"
        model.vision_model.config._attn_implementation = "eager"
        # Set patch_size from the model config for sub-patch pooling
        self.patch_size = getattr(model.config.vision_config, "patch_size", self.patch_size)
        for param in model.parameters():
            param.requires_grad = False
        if device != "cpu":
            model = model.to(device=device, dtype=torch.float32)
        self._cache[device] = (model, processor)
        return model, processor

    def prepare(self, *, prompt: str, device: str) -> PreparedAdapter:
        import torch
        import torch.nn.functional as F

        model, processor = self._load(device)
        text_inputs = processor(text=[prompt], return_tensors="pt", padding=True)
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)

        scorer = _HFCLIPScorer(model, text_features)

        def prepare_image(image: Image.Image) -> tuple[Any, Image.Image]:
            payload = processor(images=image, return_tensors="pt")
            pixel_values = payload["pixel_values"]
            image_mean = processor.image_processor.image_mean
            image_std = processor.image_processor.image_std

            display_tensor = pixel_values[0].detach().cpu().clone()
            for channel, (mean, std) in enumerate(zip(image_mean, image_std, strict=True)):
                display_tensor[channel] = (display_tensor[channel] * std) + mean
            display_image = tensor_image_to_pil(display_tensor.numpy())
            return pixel_values, display_image

        return PreparedAdapter(
            scorer=scorer,
            prepare_image=prepare_image,
            metadata={
                "model_id": self.model_id,
                "family": "huggingface-clip",
                "backend_mode": "attnlrp-efficient-clip",
            },
        )