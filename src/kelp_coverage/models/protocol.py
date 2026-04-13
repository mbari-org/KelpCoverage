from typing import Callable, Dict, List, Protocol, Tuple
import numpy as np
import torch

class SegmentationModel(Protocol):
    img_size: int
    def preprocess(
        self, slices: List[np.ndarray]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        ...

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        ...

    def decode(
        self,
        embedding: torch.Tensor,
        points_xy: np.ndarray,
        original_size: Tuple[int, int],
        input_size: Tuple[int, int],
    ) -> torch.Tensor:
        ...

_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator

def load_model(name: str, checkpoint: str, device: str) -> SegmentationModel:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](checkpoint, device)
