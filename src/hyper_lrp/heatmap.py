from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_signed_heatmap(heatmap: np.ndarray) -> np.ndarray:
    signed = np.asarray(heatmap, dtype=np.float32)
    if signed.ndim != 2:
        raise ValueError(f"heatmap must be rank-2, got shape={signed.shape}")

    abs_values = np.abs(signed)
    scale = float(np.percentile(abs_values, 99.5))
    if scale <= 1e-12:
        return np.zeros_like(signed, dtype=np.float32)
    normalized = signed / scale
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def pool_to_patches(heatmap: np.ndarray, patch_size: int) -> np.ndarray:
    """Average a pixel-level heatmap within each non-overlapping patch block.

    ViT models produce patch-level features, so the pixel-level gradient through
    the Conv2d patch embedding has sub-patch noise.  This helper averages each
    patch_size×patch_size block and upsamples back to the original spatial size,
    removing the speckle while preserving patch-level structure.
    """
    h, w = heatmap.shape
    ph = (h // patch_size) * patch_size
    pw = (w // patch_size) * patch_size
    cropped = heatmap[:ph, :pw]
    blocks = cropped.reshape(ph // patch_size, patch_size, pw // patch_size, patch_size)
    pooled = blocks.mean(axis=(1, 3))  # (n_patches_h, n_patches_w)
    # Upsample back to original size via nearest-neighbor repeat
    upsampled = np.repeat(np.repeat(pooled, patch_size, axis=0), patch_size, axis=1)
    # Pad back if original size wasn't evenly divisible
    if upsampled.shape != heatmap.shape:
        result = np.zeros_like(heatmap)
        result[:upsampled.shape[0], :upsampled.shape[1]] = upsampled
        return result
    return upsampled


def tensor_image_to_pil(image: np.ndarray) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"expected CHW image array, got shape={array.shape}")
    hwc = np.transpose(np.clip(array, 0.0, 1.0), (1, 2, 0))
    return Image.fromarray((hwc * 255.0).round().astype(np.uint8), mode="RGB")


def overlay_signed_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.72,
    gamma: float = 0.65,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    signed = normalize_signed_heatmap(heatmap)
    if base.shape[:2] != signed.shape:
        raise ValueError(
            f"image and heatmap size mismatch: image={base.shape[:2]} heatmap={signed.shape}"
        )

    positive = np.clip(signed, 0.0, 1.0)
    negative = np.clip(-signed, 0.0, 1.0)

    color = np.zeros_like(base)
    color[..., 0] = positive
    color[..., 1] = (positive * 0.35) + (negative * 0.45)
    color[..., 2] = negative

    mask = (np.power(np.clip(np.abs(signed), 0.0, 1.0), gamma) * alpha)[..., None]
    mask = np.clip(mask, 0.0, 1.0)
    composed = (base * (1.0 - mask)) + (color * mask)
    return Image.fromarray((np.clip(composed, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")