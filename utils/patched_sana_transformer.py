# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union
from utils.patch_sana_attention_layers import PatchedSanaTransformerBlock, patch_sana_attention_layers

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    SanaLinearAttnProcessor2_0,
)
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.models.transformers.sana_transformer import SanaTransformerBlock, SanaCombinedTimestepGuidanceEmbeddings, SanaModulatedNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PatchedSanaTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A 2D Transformer model introduced in [Sana](https://huggingface.co/papers/2410.10629) family of models.

    Args:
        in_channels (`int`, defaults to `32`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `32`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `70`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `32`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of Transformer blocks to use.
        num_cross_attention_heads (`int`, *optional*, defaults to `20`):
            The number of heads to use for cross-attention.
        cross_attention_head_dim (`int`, *optional*, defaults to `112`):
            The number of channels in each head for cross-attention.
        cross_attention_dim (`int`, *optional*, defaults to `2240`):
            The number of channels in the cross-attention output.
        caption_channels (`int`, defaults to `2304`):
            The number of channels in the caption embeddings.
        mlp_ratio (`float`, defaults to `2.5`):
            The expansion ratio to use in the GLUMBConv layer.
        dropout (`float`, defaults to `0.0`):
            The dropout probability.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in the attention layer.
        sample_size (`int`, defaults to `32`):
            The base size of the input latent.
        patch_size (`int`, defaults to `1`):
            The size of the patches to use in the patch embedding layer.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether to use elementwise affinity in the normalization layer.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value for the normalization layer.
        qk_norm (`str`, *optional*, defaults to `None`):
            The normalization to use for the query and key.
        timestep_scale (`float`, defaults to `1.0`):
            The scale to use for the timesteps.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SanaTransformerBlock", "PatchEmbed", "SanaModulatedNorm"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

    @register_to_config
    def __init__(
        self,
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
        guidance_embeds_scale: float = 0.1,
        qk_norm: Optional[str] = None,
        timestep_scale: float = 1.0,
        modified_blocks: Optional[int] = []
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
            pos_embed_type="sincos" if interpolation_scale is not None else None,
        )

        # 2. Additional condition embeddings
        if guidance_embeds:
            self.time_embed = SanaCombinedTimestepGuidanceEmbeddings(inner_dim)
        else:
            self.time_embed = AdaLayerNormSingle(inner_dim)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SanaTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # patch the blocks
        patch_sana_attention_layers(self, modified_blocks)

        # 4. Output blocks
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = SanaModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

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
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        else:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
