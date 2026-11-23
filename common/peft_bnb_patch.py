import functools
from types import SimpleNamespace
import torch
from bitsandbytes.nn import Linear4bit
from peft.tuners.lokr.layer import LoKrLayer
from peft.tuners.loha.layer import LoHaLayer
from common.torchao_nf4 import NF4Conv1x1


def _patch_get_delta_weight(layer_cls):
    original = layer_cls.get_delta_weight
    if getattr(original, '_bnb_patched', False):
        return

    @functools.wraps(original)
    def get_delta_weight(self, *args, **kwargs):
        base_layer = self.get_base_layer()
        if isinstance(base_layer, Linear4bit):
            # a bnb 4bit weight is stored packed (numel // 2, 1), but the delta is built at the dequantized
            # (out_features, in_features) size. The original only reads base_layer.weight.shape, so hand it
            # a meta tensor of the right shape instead of the packed one.
            shape = (base_layer.out_features, base_layer.in_features)
        elif isinstance(base_layer, NF4Conv1x1):
            # the torchao NF4 weight of a 1x1 conv is stored as an (out, in) matrix, since NF4Tensor
            # has no 4D views, but the conv adapter needs its delta at the (out, in, 1, 1) size
            shape = (base_layer.out_channels, base_layer.in_channels, 1, 1)
        else:
            return original(self, *args, **kwargs)
        shim = SimpleNamespace(weight=torch.empty(*shape, device='meta'))
        self.get_base_layer = lambda: shim
        try:
            return original(self, *args, **kwargs)
        finally:
            del self.get_base_layer

    get_delta_weight._bnb_patched = True
    layer_cls.get_delta_weight = get_delta_weight


def patch_lycoris_for_bnb_4bit():
    """peft's LoKr and LoHa reshape their delta to the shape of the stored weight of a bnb 4bit layer or of an
    NF4 1x1 conv, which is not the shape of the dequantized weight, so that fails."""
    _patch_get_delta_weight(LoKrLayer)
    _patch_get_delta_weight(LoHaLayer)
