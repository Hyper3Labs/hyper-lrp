from __future__ import annotations

from hyper_lrp.adapters import HuggingFaceCLIPAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, object] = {}

    def register(self, name: str, adapter: object) -> None:
        self._adapters[name] = adapter

    def get(self, name: str) -> object:
        try:
            return self._adapters[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters))
            raise KeyError(f"Unknown adapter '{name}'. Available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._adapters)


def create_default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register("hf-clip", HuggingFaceCLIPAdapter())
    return registry