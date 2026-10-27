import functools
from types import SimpleNamespace
import torch
from bitsandbytes.nn import Linear4bit
from peft.tuners.lokr.layer import LoKrLayer
from peft.tuners.loha.layer import LoHaLayer


def _patch_get_delta_weight(layer_cls):
    original = layer_cls.get_delta_weight
    if getattr(original, '_bnb_patched', False):
        return

    @functools.wraps(original)
    def get_delta_weight(self, *args, **kwargs):
        base_layer = self.get_base_layer()
        if not isinstance(base_layer, Linear4bit):
            return original(self, *args, **kwargs)
        # a bnb 4bit weight is stored packed (numel // 2, 1), but the delta is built at the dequantized
        # (out_features, in_features) size. The original only reads base_layer.weight.shape, so hand it
        # a meta tensor of the right shape instead of the packed one.
        shim = SimpleNamespace(weight=torch.empty(base_layer.out_features, base_layer.in_features, device='meta'))
        self.get_base_layer = lambda: shim
        try:
            return original(self, *args, **kwargs)
        finally:
            del self.get_base_layer

    get_delta_weight._bnb_patched = True
    layer_cls.get_delta_weight = get_delta_weight


def patch_lycoris_for_bnb_4bit():
    """peft's LoKr and LoHa reshape their delta to the packed shape of a bnb 4bit weight, which fails."""
    _patch_get_delta_weight(LoKrLayer)
    _patch_get_delta_weight(LoHaLayer)
