from diffusers import SanaTransformer2DModel
from diffusers.models.attention_processor import Attention, AttnProcessor

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