from diffusers import PixArtTransformer2DModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from typing import Optional
import torch.nn as nn
import torch

class ResidualConv(nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        kernel_size = 8 - 1
        padding = (kernel_size + 1) // 2 - 1
        self.out_net = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=kernel_size, padding=padding),
        )
        self.out_alpha = nn.Parameter(torch.tensor(0.1))
        self.cross_attention_dim = cross_attention_dim

        for module in self.out_net:
            if hasattr(module, 'weight'):
                nn.init.dirac_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shape = x.shape
        y = torch.reshape(x, (4, -1, self.cross_attention_dim))
        z = self.out_alpha * self.out_net(y)
        y = y + z
        return y.reshape(shape)

class ExpandedAttention(Attention):
    def __init__(self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False):
        super().__init__(query_dim,
                       cross_attention_dim,
                       heads,
                       kv_heads,
                       dim_head,
                       dropout,
                       bias,
                       upcast_attention,
                       upcast_softmax,
                       cross_attention_norm,
                       cross_attention_norm_num_groups,
                       qk_norm,
                       added_kv_proj_dim,
                       added_proj_bias,
                       norm_num_groups,
                       spatial_norm_dim,
                       out_bias,
                       scale_qk,
                       only_cross_attention,
                       eps,
                       rescale_output_factor,
                       residual_connection,
                       _from_deprecated_attn_block,
                       processor,
                       out_dim,
                       out_context_dim,
                       context_pre_only,
                       pre_only,
                       elementwise_affine,
                       is_causal)
        
        # convolution layers
        self.conv_layers = nn.ModuleList([ResidualConv(self.cross_attention_dim) for _ in range(1)])

    
    @classmethod
    def from_attention_block(cls, other : Attention,
                             dim,
                             num_attention_heads,
                             attention_head_dim,
                             dropout,
                             attention_bias,
                             cross_attention_dim,
                             upcast_attention,
                             attention_out_bias):
        attention = cls(query_dim=dim,
                        cross_attention_dim=cross_attention_dim,
                        heads=num_attention_heads,
                        dim_head=attention_head_dim,
                        dropout=dropout,
                        bias=attention_bias,
                        upcast_attention=upcast_attention,
                        out_bias=attention_out_bias)

        # copy the weigths
        attention.load_state_dict(other.state_dict(), strict=False)
        return attention
        

    def forward(self, hidden_states, encoder_hidden_states = None, attention_mask = None, **cross_attention_kwargs):
        hidden_states = super().forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)

        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class PixArtTransformer2DModelWithResNet(PixArtTransformer2DModel):
    def __init__(        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = 8,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        use_additional_conditions: Optional[bool] = None,
        caption_channels: Optional[int] = None,
        attention_type: Optional[str] = "default"):
        super().__init__(num_attention_heads=num_attention_heads,
                         attention_head_dim=attention_head_dim,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         num_layers=num_layers,
                         dropout=dropout,
                         norm_num_groups=norm_num_groups,
                         cross_attention_dim=cross_attention_dim,
                         attention_bias=attention_bias,
                         sample_size=sample_size,
                         patch_size=patch_size,
                         activation_fn=activation_fn,
                         num_embeds_ada_norm=num_embeds_ada_norm,
                         upcast_attention=upcast_attention,
                         norm_type=norm_type,
                         norm_elementwise_affine=norm_elementwise_affine,
                         norm_eps=norm_eps,
                         interpolation_scale=interpolation_scale,
                         use_additional_conditions=use_additional_conditions,
                         caption_channels=caption_channels,
                         attention_type=attention_type)
        self.expand()
        

    def expand(self):
        transformer_blocks = self.transformer_blocks
        for block in transformer_blocks:
            block.attn1 = ExpandedAttention.from_attention_block(block.attn1,
                                                                block.dim,
                                                                block.num_attention_heads,
                                                                block.attention_head_dim,
                                                                block.dropout,
                                                                block.attention_bias,
                                                                block.cross_attention_dim,
                                                                block.attn1.upcast_attention,
                                                                self.config.attention_bias)
            block.attn2 = ExpandedAttention.from_attention_block(block.attn2,
                                                                block.dim,
                                                                block.num_attention_heads,
                                                                block.attention_head_dim,
                                                                block.dropout,
                                                                block.attention_bias,
                                                                block.cross_attention_dim,
                                                                block.attn2.upcast_attention,
                                                                self.config.attention_bias)

def expand_pixart_sigma_transformer(transformer):
    state_dict = transformer.state_dict()
    transformer = PixArtTransformer2DModelWithResNet.from_config(transformer.config)
    transformer.load_state_dict(state_dict, strict=False)
    return transformer