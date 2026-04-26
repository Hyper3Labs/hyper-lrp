from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(slots=True)
class LRPResult:
    adapter_name: str
    prompt: str
    score: float
    heatmap: np.ndarray
    display_image: Image.Image
    overlay_image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        heatmap_path = output_path / "heatmap.npy"
        display_path = output_path / "input.png"
        overlay_path = output_path / "overlay.png"
        meta_path = output_path / "result.json"

        np.save(heatmap_path, self.heatmap.astype(np.float32))
        self.display_image.save(display_path)
        self.overlay_image.save(overlay_path)
        meta_path.write_text(
            __import__("json").dumps(
                {
                    "adapter_name": self.adapter_name,
                    "prompt": self.prompt,
                    "score": self.score,
                    "metadata": self.metadata,
                },
                indent=2,
                sort_keys=True,
            )
        )

        return {
            "heatmap": heatmap_path,
            "input": display_path,
            "overlay": overlay_path,
            "metadata": meta_path,
        }