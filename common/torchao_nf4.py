import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import NF4Tensor, to_nf4
from torchao.quantization.quantize_.workflows.nf4.nf4_tensor import linear_nf4


class NF4Conv1x1(nn.Conv2d):
    """A 1x1 Conv2d whose weight is a torchao NF4Tensor.

    bitsandbytes only quantizes nn.Linear, which leaves the 1x1 convolutions of SANA's Mix-FFN (about
    half of the weights of the transformer) in bf16. A 1x1 conv is a matmul over the channels, so it
    runs as a linear on the channels last view. The module stays a Conv2d so that peft attaches its
    usual convolution adapter to it. The weight is stored as the 2D (out, in) matrix NF4Tensor needs.
    """

    def forward(self, input):
        out = linear_nf4(input.movedim(1, -1), self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out.movedim(-1, 1)


def _is_quantizable(module):
    return (type(module) is nn.Conv2d and module.kernel_size == (1, 1) and module.stride == (1, 1)
            and module.padding == (0, 0) and module.groups == 1 and module.dilation == (1, 1)
            and module.padding_mode == 'zeros')


def quantize_conv1x1_nf4(model, block_size=64, scaler_block_size=256):
    """Replace every 1x1 Conv2d of the model by an NF4Conv1x1. Returns the number of layers converted.

    The weights are quantized one layer at a time on the device they already live on, so the bf16
    copy of the layer is released as soon as its NF4 version exists.
    """
    converted = 0
    for name, module in list(model.named_modules()):
        if not _is_quantizable(module):
            continue
        weight = module.weight.data
        if weight.numel() % block_size != 0:
            continue
        # NF4 double quantizes its per block scalers in groups of scaler_block_size, so that has to
        # divide the number of blocks of the layer
        layer_scaler_block_size = math.gcd(weight.numel() // block_size, scaler_block_size)
        matrix = weight.reshape(weight.shape[0], weight.shape[1])
        module.weight = nn.Parameter(to_nf4(matrix, block_size, layer_scaler_block_size), requires_grad=False)
        module.__class__ = NF4Conv1x1
        converted += 1
    return converted
