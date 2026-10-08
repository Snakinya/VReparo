"""Paper identity-centered, per-head affine ridge fitting."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from transformers.cache_utils import DynamicCache
from .cache import layer_kv

@dataclass
class ValueRidgeStatistics:
    """Sufficient statistics for independent per-layer, per-KV-head ridge fits."""

    count: int
    sum_x: torch.Tensor
    sum_y: torch.Tensor
    sum_xx: torch.Tensor
    sum_xy: torch.Tensor

    @classmethod
    def empty(cls, *, layers: int, kv_heads: int, head_dim: int) -> "ValueRidgeStatistics":
        shape = (layers, kv_heads, head_dim)
        matrix_shape = (layers, kv_heads, head_dim, head_dim)
        return cls(
            count=0,
            sum_x=torch.zeros(shape, dtype=torch.float64),
            sum_y=torch.zeros(shape, dtype=torch.float64),
            sum_xx=torch.zeros(matrix_shape, dtype=torch.float64),
            sum_xy=torch.zeros(matrix_shape, dtype=torch.float64),
        )

    def update(
        self,
        producer_cache: object,
        receiver_cache: object,
        num_layers: int,
        *,
        max_tokens: int | None = None,
        component: str = "value",
    ) -> None:
        """Accumulate aligned K- or V-token pairs from one paired prefill."""
        if component not in {"key", "value"}:
            raise ValueError(f"Unsupported cache component: {component!r}")
        tokens_added: int | None = None
        for layer in range(num_layers):
            producer_k, producer_v = layer_kv(producer_cache, layer)
            receiver_k, receiver_v = layer_kv(receiver_cache, layer)
            producer_tensor = producer_k if component == "key" else producer_v
            receiver_tensor = receiver_k if component == "key" else receiver_v
            if producer_tensor.shape != receiver_tensor.shape:
                raise ValueError(
                    f"{component.title()} shape mismatch at layer {layer}: "
                    f"{producer_tensor.shape} vs {receiver_tensor.shape}"
                )
            if producer_tensor.ndim != 4 or producer_tensor.shape[0] != 1:
                raise ValueError("Ridge mapper expects an unpadded batch of size one")
            # [batch, heads, tokens, dim] -> [tokens, heads, dim].  A fixed
            # stride makes calibration length-independent and deterministic.
            x = producer_tensor[0].transpose(0, 1)
            y = receiver_tensor[0].transpose(0, 1)
            if max_tokens is not None and x.shape[0] > max_tokens:
                stride = max(1, x.shape[0] // max_tokens)
                x, y = x[::stride][:max_tokens], y[::stride][:max_tokens]
            if tokens_added is None:
                tokens_added = int(x.shape[0])
            elif tokens_added != int(x.shape[0]):
                raise ValueError("Cache layers have inconsistent token lengths")
            x = x.detach().to(device="cpu", dtype=torch.float64)
            y = y.detach().to(device="cpu", dtype=torch.float64)
            self.sum_x[layer] += x.sum(dim=0)
            self.sum_y[layer] += y.sum(dim=0)
            self.sum_xx[layer] += torch.einsum("thd,the->hde", x, x)
            self.sum_xy[layer] += torch.einsum("thd,the->hde", x, y)
        self.count += tokens_added or 0


@dataclass(frozen=True)
class ValueRidgeMapper:
    """Affine producer-value to receiver-value mapper, fitted independently per head."""

    weights: torch.Tensor  # [layers, kv_heads, source_dim, receiver_dim]
    bias: torch.Tensor  # [layers, kv_heads, receiver_dim]
    ridge: float
    calibration_tokens: int

    @classmethod
    def fit(
        cls,
        statistics: ValueRidgeStatistics,
        *,
        ridge: float,
        identity_centered: bool = False,
    ) -> "ValueRidgeMapper":
        if statistics.count < 2:
            raise ValueError("At least two calibration tokens are required")
        if ridge < 0:
            raise ValueError("Ridge penalty must be non-negative")
        count = float(statistics.count)
        mean_x = statistics.sum_x / count
        mean_y = statistics.sum_y / count
        xx = statistics.sum_xx - count * torch.einsum("lhd,lhe->lhde", mean_x, mean_x)
        xy = statistics.sum_xy - count * torch.einsum("lhd,lhe->lhde", mean_x, mean_y)
        dimension = xx.shape[-1]
        eye = torch.eye(dimension, dtype=xx.dtype).view(1, 1, dimension, dimension)
        # Standard ridge shrinks W toward zero.  For same-foundation model
        # variants, an identity-centred penalty is a better controlled
        # alternative: it asks the mapper to make the smallest correction to
        # the already compatible producer representation.
        target = xy + ridge * eye if identity_centered else xy
        weights = torch.linalg.solve(xx + ridge * eye, target)
        bias = mean_y - torch.einsum("lhd,lhde->lhe", mean_x, weights)
        return cls(
            weights=weights.to(dtype=torch.float32),
            bias=bias.to(dtype=torch.float32),
            ridge=float(ridge),
            calibration_tokens=statistics.count,
        )

    def transform_cache(self, cache: object, num_layers: int) -> DynamicCache:
        """Return a fresh cache with original keys and mapped values.

        Mapping is applied before the existing cache-mixing boundary.  The
        receiver still recomputes the final prompt token and owns all decoded
        tokens, exactly as in the raw-reuse reference path.
        """
        if self.weights.shape[0] != num_layers:
            raise ValueError(f"Mapper has {self.weights.shape[0]} layers, expected {num_layers}")
        mapped = DynamicCache()
        for layer in range(num_layers):
            key, value = layer_kv(cache, layer)
            transformed = self.transform_tensor(value, layer)
            mapped.update(key, transformed, layer)
        return mapped

    def transform_tensor(self, tensor: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply one layer's per-head affine map to an explicit K or V tensor."""
        weight = self.weights[layer].to(device=tensor.device, dtype=torch.float32)
        bias = self.bias[layer].to(device=tensor.device, dtype=torch.float32)
        if tensor.shape[1] != weight.shape[0] or tensor.shape[-1] != weight.shape[1]:
            raise ValueError(f"Mapper/tensor shape mismatch at layer {layer}")
        transformed = torch.einsum("bhtd,hde->bhte", tensor.float(), weight)
        return (
            transformed + bias.unsqueeze(0).unsqueeze(2)
        ).to(dtype=tensor.dtype).contiguous()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "weights": self.weights,
            "bias": self.bias,
            "ridge": self.ridge,
            "calibration_tokens": self.calibration_tokens,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "ValueRidgeMapper":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported value mapper schema")
        return cls(
            weights=payload["weights"],
            bias=payload["bias"],
            ridge=float(payload["ridge"]),
            calibration_tokens=int(payload["calibration_tokens"]),
        )
