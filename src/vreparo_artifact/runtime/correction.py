"""Paper Value-only interpolation, preserving intermediate dtype rounding."""
from dataclasses import dataclass
import torch
from transformers.cache_utils import DynamicCache
from .cache import layer_kv

@dataclass
class ValueCorrection:
    payload: dict
    alpha: float

    def transform_tensor(self, value, layer):
        if self.alpha == 0 or layer not in self.payload['active_layers']:
            return value
        x = value.float()
        p = self.payload
        bias = p['bias'][layer].to(x.device)
        if p['kind'] == 'dense':
            aligned = torch.einsum('bhtd,hde->bhte', x, p['weights'][layer].to(x.device))
            aligned = (aligned + bias[None, :, None, :]).to(value.dtype)
            delta = aligned.float() - x
        else:
            coeff = torch.einsum('bhtd,hdr->bhtr', x, p['left'][layer].to(x.device))
            delta = torch.einsum('bhtr,hre->bhte', coeff, p['right'][layer].to(x.device))
            # Match dense's intermediate mapped-V dtype before blending.
            aligned = (x + delta + bias[None, :, None, :]).to(value.dtype)
            delta = aligned.float() - x
        return (x + self.alpha * delta).to(value.dtype).contiguous()

    def transform_cache(self, cache, num_layers):
        out = DynamicCache()
        for layer in range(num_layers):
            k, v = layer_kv(cache, layer)
            out.update(k, self.transform_tensor(v, layer), layer)
        return out

    def to(self, device):
        return ValueCorrection({k: v.to(device) if torch.is_tensor(v) else v
                                for k, v in self.payload.items()}, self.alpha)
