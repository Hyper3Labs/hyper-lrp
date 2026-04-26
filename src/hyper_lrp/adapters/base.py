from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image


@dataclass(slots=True)
class PreparedAdapter:
    scorer: Any
    prepare_image: Callable[[Image.Image], tuple[Any, Image.Image]]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    name: str

    @staticmethod
    def resolve_device(device: str | None) -> str:
        if device:
            return device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @abstractmethod
    def ensure_patched(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, *, prompt: str, device: str) -> PreparedAdapter:
        raise NotImplementedError


class GenericImageTextAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        name: str,
        patch_callback: Callable[[], None],
        prepare_callback: Callable[[str, str], PreparedAdapter],
    ) -> None:
        self.name = name
        self._patch_callback = patch_callback
        self._prepare_callback = prepare_callback
        self._patched = False

    def ensure_patched(self) -> None:
        if not self._patched:
            self._patch_callback()
            self._patched = True

    def prepare(self, *, prompt: str, device: str) -> PreparedAdapter:
        return self._prepare_callback(prompt, device)