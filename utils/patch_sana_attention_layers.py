from diffusers import SanaTransformer2DModel
from torch import nn
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    SanaLinearAttnProcessor2_0,
)
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.transformers.sana_transformer import GLUMBConv
from diffusers.models.attention_processor import AttnProcessor2_0
import torch

class PatchedSanaTransformerBlock(nn.Module):
    r"""
    Transformer block introduced in [Sana](https://huggingface.co/papers/2410.10629).
    """

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 2.5,
        qk_norm: Optional[str] = None,
    ) -> None:
        super().__init__()

        # 1. Self Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads if qk_norm is not None else None,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            processor=AttnProcessor()
        )

        # 2. Cross Attention
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.attn2 = Attention(
                query_dim=dim,
                qk_norm=qk_norm,
                kv_heads=num_cross_attention_heads if qk_norm is not None else None,
                cross_attention_dim=cross_attention_dim,
                heads=num_cross_attention_heads,
                dim_head=cross_attention_head_dim,
                dropout=dropout,
                bias=True,
                out_bias=attention_out_bias,
                processor=AttnProcessor(),
            )

        # 3. Feed-forward
        self.ff = GLUMBConv(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states

def unfreeze_sana_blocks(transformer, layers : list[int]):
    for param in transformer.parameters():
        param.requires_grad = False
    for idx in layers:
        block = transformer.transformer_blocks[idx]
        for param in block.parameters():
            param.requires_grad = True

def patch_sana_attention_layers(transformer, layers : list[int]):
    # freeze the whole model first
    for param in transformer.parameters():
        param.requires_grad = False

    for idx in layers:
        block = transformer.transformer_blocks[idx]
        block.attn1.set_processor(AttnProcessor2_0())
        block.attn2.set_processor(AttnProcessor2_0())
        
        # Unfreeze the new block
        for param in block.parameters():
            param.requires_grad = True

        transformer.transformer_blocks[idx] = block
    
    modified_blocks = transformer.config.modified_blocks
    for l in layers:
        if not l in modified_blocks:
            modified_blocks.append(l)
    transformer.register_to_config(modified_blocks = modified_blocks)


