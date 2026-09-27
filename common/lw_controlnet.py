"""Wiring of the lightweight_controlnet control path into the trainers.

The submodule itself knows nothing about this repo: it hooks the layers a config names so that the
inverse real FFT of a sparse complex half-spectrum is added to each one's output, part of that
spectrum written by learned coefficients and part by a projection of the reference image features,
and hands back a FourierControl. Everything model specific - which config the yaml describes, how a PIL
image becomes the tensor the encoder wants, and how a checkpoint is written so that
infer_krea2_controlnet.py can read it back - lives here, so common/trainer.py stays as generic as it
was.

The path is attached as forward hooks, not by replacing modules, so the model being trained keeps
its module tree and its state dict: a peft adapter saves as an ordinary peft adapter with nothing to
undo first, and the control path lands next to it in fourier_control.pt.
"""
import os
from contextlib import contextmanager

import torch

from lightweight_controlnet import (
    FourierControl,
    FourierControlConfig,
    ReferenceEncoder,
    ReferenceFeatures,
    apply_fourier_control,
)

# hidden size of the usual dinov2 checkpoints, so the yaml only has to name the model
REFERENCE_DIMS = {
    'facebook/dinov2-small': 384,
    'facebook/dinov2-base': 768,
    'facebook/dinov2-large': 1024,
    'facebook/dinov2-giant': 1536,
}

# name under which the control path is hung on the model being trained. It has to be a module of
# that model for DDP to synchronize its gradients and for the optimizer to see it at all, and it
# must not contain 'lora_': that is the substring peft filters its state dict with, and this is not
# part of the adapter.
CONTROL_ATTRIBUTE = 'fourier_control'


def reference_dim_for(model_name: str, fallback: int = 768) -> int:
    if model_name in REFERENCE_DIMS:
        return REFERENCE_DIMS[model_name]
    for name, dim in REFERENCE_DIMS.items():
        if model_name.endswith(name.split('/')[-1]):
            return dim
    return fallback


def build_config(params) -> FourierControlConfig:
    """The FourierControlConfig a training config describes."""
    return FourierControlConfig(
        target_modules=params.lw_controlnet_target_modules,
        adapter_name=params.lw_controlnet_adapter_name,
        reference_model=params.lw_controlnet_reference_model,
        reference_dim=(params.lw_controlnet_reference_dim
                       if params.lw_controlnet_reference_dim != None
                       else reference_dim_for(params.lw_controlnet_reference_model)),
        cond_dim=params.lw_controlnet_cond_dim,
        pool_grid=params.lw_controlnet_pool_grid,
        n_frequency=params.lw_controlnet_n_frequency,
        frequency_ratio=params.lw_controlnet_frequency_ratio,
        n_reference=params.lw_controlnet_n_reference,
        reference_ratio=params.lw_controlnet_reference_ratio,
        spectrum_scaling=params.lw_controlnet_spectrum_scaling,
        reference_scaling=params.lw_controlnet_reference_scaling,
        ifft_norm=params.lw_controlnet_ifft_norm,
        random_loc_seed=params.lw_controlnet_random_loc_seed,
        dtype=params.lw_controlnet_dtype,
    )


def attach_control(model, config, device=None) -> FourierControl:
    """Hook the layers of model and register the control path as one of its submodules.

    The hooks leave the model itself untouched, so all the trainable weights live in the
    FourierControl. It is hung on the model here so that it is covered by the optimizer, by
    accelerate's prepare and by the gradient synchronization of a multi gpu run, exactly like the
    rest of the trainable weights.
    """
    control = apply_fourier_control(model, config, device=device)
    model.add_module(CONTROL_ATTRIBUTE, control)
    return control


@contextmanager
def detached_control(model):
    """Take the control path off the model for the duration of the block.

    Only saving needs this, and for one reason: the path is hung on the model so that the optimizer
    and DDP can see its weights, which also puts it in whatever state dict the model writes. A peft
    save drops it on its own - peft filters on the 'lora_' substring, which none of these parameter
    names carry - but a plain diffusers save does not, and would write the control path into the
    model's own checkpoint. Detaching covers both.

    The hooks stay in place and the weights stay where they are; only the model's reference to them
    goes away, so a forward pass inside this block still runs the control path.
    """
    control = getattr(model, CONTROL_ATTRIBUTE, None)
    if control is None:
        yield
        return
    delattr(model, CONTROL_ATTRIBUTE)
    try:
        yield
    finally:
        model.add_module(CONTROL_ATTRIBUTE, control)


def save_control(control, directory):
    """fourier_control.pt plus fourier_control_config.json, next to the peft adapter."""
    os.makedirs(directory, exist_ok=True)
    control.save_pretrained(directory)


def load_control(control, directory):
    control.load_pretrained(directory)


def to_reference_tensor(image):
    """A PIL image (or a tensor already) as the (3, h, w) float in [0, 1] the encoder expects.

    No resize here: preprocess_reference resolves its own aspect ratio preserving patch grid from
    the source resolution, which is the whole point of not running the reference through the
    aspect ratio buckets of the diffusion path.
    """
    if torch.is_tensor(image):
        tensor = image.float()
        return tensor / 255.0 if tensor.max() > 1.5 else tensor
    from torchvision.transforms.functional import pil_to_tensor
    return pil_to_tensor(image.convert('RGB')).float() / 255.0


def build_encoder(config, device=None, dtype=torch.bfloat16):
    encoder = ReferenceEncoder(config)
    if device is not None:
        encoder = encoder.to(device=device)
    return encoder.to(dtype=dtype)


def features_to(features: ReferenceFeatures, device) -> ReferenceFeatures:
    """ReferenceFeatures.to only follows the tensors it owns, the mask included."""
    return ReferenceFeatures(
        patch_tokens=features.patch_tokens.to(device),
        cls_token=features.cls_token.to(device),
        grid=features.grid,
        aspect_ratio=features.aspect_ratio.to(device),
        padding_mask=None if features.padding_mask is None else features.padding_mask.to(device),
    )


@contextmanager
def frozen_diffusion_path(model, control):
    """Run the block with every weight but the control path's frozen.

    Unlike the cross attention architecture this replaced, there is no startup trap in it. That one
    reached the output through peft's lora_B, which starts at zero, so freezing the lora on every
    batch that carried a reference left the whole control path with exactly zero gradient and stalled
    a fresh run. Here the path reaches the output directly - it is added to the layer's output, not
    routed through anything the lora owns - so its gradient does not depend on any other weight being
    nonzero, and this can be turned on from step 0.

    requires_grad is what is toggled, so the forward pass has to be built inside the block - and the
    backward pass has to stay inside it too, for the same reason batch_backward keeps it inside the
    reference context.
    """
    control_parameters = {id(parameter) for parameter in control.parameters()}
    frozen = []
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in control_parameters:
            parameter.requires_grad_(False)
            frozen.append(parameter)
    try:
        yield
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
