from diffusers import PixArtTransformer2DModel
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.configuration_utils import register_to_config
from transformers import PretrainedConfig


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, projector_size, output_size):
        super().__init__()
        self.fc = torch.nn.Sequential(
            nn.Linear(input_size, projector_size),
            nn.SiLU(),
            nn.Linear(projector_size, projector_size),
            nn.SiLU(),
            nn.Linear(projector_size, output_size),
        )

    def forward(self, x):
        return self.fc(x)

class REPAPixArtTransformerModel(PixArtTransformer2DModel):
    r"""
    Applies REPA training technique to the PixartSigma Transformer module
    https://arxiv.org/abs/2410.06940
    """
    @register_to_config
    def __init__(
        self,
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
        attention_type: Optional[str] = "default",
        repa_depth: int = 4,
        repa_encoder_hidden_size = 768,
        repa_projector_dim = 2048,
    ):
        super().__init__(num_attention_heads,
                         attention_head_dim,
                         in_channels,
                         out_channels,
                         num_layers,
                         dropout,
                         norm_num_groups,
                         cross_attention_dim,
                         attention_bias,
                         sample_size,
                         patch_size,
                         activation_fn,
                         num_embeds_ada_norm,
                         upcast_attention,
                         norm_type,
                         norm_elementwise_affine,
                         norm_eps,
                         interpolation_scale,
                         use_additional_conditions,
                         caption_channels,
                         attention_type)
        
        # we need to apply repa mlps to the first repa_depth layers of the transformer
        self.repa_depth = repa_depth
        self.repa_projector_dim = repa_projector_dim
        self.repa_encoder_hidden_size = repa_encoder_hidden_size
        self.repa_mlps = ThreeLayerMLP(cross_attention_dim, 
                                                      repa_projector_dim, 
                                                      repa_encoder_hidden_size).to(self.device)
        self.repa_proj = None

    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True):
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        repa_index = 0
        repa_proj = None
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    None,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )
            
            # do REPA projections only in training
            #if self.training and self.repa_index < self.repa_depth:
            if repa_index == self.repa_depth:
                N, T, D = hidden_states.shape
                repa_proj = self.repa_mlps \
                    (hidden_states.reshape(-1, D))
                
                repa_proj = repa_proj.reshape(N, T, -1)
                self.repa_proj = repa_proj
            repa_index = repa_index + 1

        # 3. Output
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def patch_pixart_sigma_transformer(transformer,
                                   repa_depth,
                                   repa_encoder_hidden_size,
                                   repa_projector_dim,
                                   cross_attention_dim):
    transformer.repa_depth = repa_depth
    transformer.repa_encoder_hidden_size = repa_encoder_hidden_size
    transformer.repa_projector_dim = repa_projector_dim
    transformer.repa_mlps = ThreeLayerMLP(cross_attention_dim, 
                                                    repa_projector_dim, 
                                                    repa_encoder_hidden_size)
    transformer.repa_projections = []
    transformer.save_pretrained('repa_test')

if __name__ == '__main__':
    mlp = ThreeLayerMLP(1152, 2048, 768)