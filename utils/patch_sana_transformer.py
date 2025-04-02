from diffusers import SanaTransformer2DModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.transformers.sana_transformer import SanaCombinedTimestepGuidanceEmbeddings
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm

class SanaTransformerFullAttentionModel(SanaTransformer2DModel):
    def __init__(self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        guidance_embeds: bool = False,
        qk_norm: Optional[str] = None):
        super().__init__(in_channels,
                         out_channels,
                         num_attention_heads,
                         attention_head_dim,
                         num_layers,
                         num_cross_attention_heads,
                         cross_attention_head_dim,
                         cross_attention_dim,
                         caption_channels,
                         mlp_ratio,
                         dropout,
                         attention_bias,
                         sample_size,
                         patch_size,
                         norm_elementwise_affine,
                         norm_eps,
                         interpolation_scale,
                         guidance_embeds,
                         qk_norm)
        patch_sana_transformer(self)

def patch_sana_transformer(transformer : SanaTransformer2DModel):
    dim = transformer.config.cross_attention_dim
    num_attention_heads = transformer.config.num_attention_heads
    attention_head_dim = transformer.config.attention_head_dim
    num_attention_heads = transformer.config.num_attention_heads
    qk_norm = transformer.config.qk_norm
    dropout = transformer.config.dropout
    cross_attention_dim = transformer.config.cross_attention_dim
    num_cross_attention_heads = transformer.config.num_cross_attention_heads
    cross_attention_head_dim = transformer.config.cross_attention_head_dim

    # replace linear attention with regular attention
    for transformer_block in transformer.transformer_blocks:
        transformer_block.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=True,
            cross_attention_dim=None,
            processor=AttnProcessor()
        )

        transformer_block.attn2 = Attention(
            query_dim=dim,
            qk_norm=qk_norm,
            kv_heads=num_cross_attention_heads,
            cross_attention_dim=cross_attention_dim,
            heads=num_cross_attention_heads,
            dim_head=cross_attention_head_dim,
            dropout=dropout,
            bias=True,
            out_bias=True,
            processor=AttnProcessor(),
        )
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total parameters in U-Net: {total_params}")

def change_sample_size(transformer : SanaTransformerFullAttentionModel):
    # 1. Patch Embedding
    transformer.config.sample_size = 32
    sample_size = transformer.config.sample_size
    patch_size = transformer.config.patch_size
    in_channels = transformer.config.in_channels
    num_attention_heads = transformer.config.num_attention_heads
    attention_head_dim = transformer.config.attention_head_dim
    inner_dim = num_attention_heads * attention_head_dim
    interpolation_scale = transformer.config.interpolation_scale
    transformer.patch_embed = PatchEmbed(
        height=sample_size,
        width=sample_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=inner_dim,
        interpolation_scale=interpolation_scale,
        pos_embed_type="sincos" if interpolation_scale is not None else None,
    )

def create_sana_transformer() -> SanaTransformer2DModel:
    config = SanaTransformer2DModel.load_config('Efficient-Large-Model/Sana_600M_1024px_diffusers', subfolder='transformer')
    transformer = SanaTransformer2DModel.from_config(config)
    patch_sana_transformer(transformer)
    return transformer