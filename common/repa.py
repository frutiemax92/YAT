import torch

from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from transformers.utils import PushToHubMixin
from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel
from diffusers.models.modeling_utils import ModelMixin
import os
import json

@dataclass
class RepaConfig:
    """
    The code style is inspired from PEFT.
    This config indicates which modules to apply REPA.
    The modules append to repa_features list on the new model when being called.
    """
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with FourierFT."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )

    target_shape : int = field(
        default=768
    )

    hidden_shape : list[int] = field(
        default=None,
        metadata={
            "help": (
                "This indicates the number of elements that compose the output of the target_modules"
                "For SANA, this is 1152 elements for example on a transformer block"
            )
        }
    )

class RepaMLP(torch.nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, out_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class RepaModule(torch.nn.Module):
    def __init__(
            self,
            base_module : torch.nn.Module,
            in_size : int,
            out_size : int
    ):
        super().__init__()
        self.zs = None
        self.base_module = base_module
        self.proj_mlp = RepaMLP(
            in_size,
            out_size,
            out_size
        )
    
    def forward(self,
                *args,
                **kwargs):
        hidden_states = self.base_module(*args, **kwargs)
        self.zs = self.proj_mlp(hidden_states)
        return hidden_states

class RepaModel(ModelMixin):
    """
    The RepaModel consists of the base model + the projection layers for REPA
    """
    def __init__(
            self,
            model: PreTrainedModel,
            repa_config: RepaConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.repa_states = []
        self.repa_config = repa_config
        self.parent_modules = {} # Store parent modules and original module names

        self._patch_base_model()

    def _patch_base_model(
            self
    ):
        """ This adds the REPA projection layers 
        """
        out_size = self.repa_config.target_shape
        for idx, module_name in enumerate(self.repa_config.target_modules):
            hidden_shape = self.repa_config.hidden_shape[idx]
            
            # Use a non-recursive approach to replace the module
            module = self.model.get_submodule(module_name)
            patched_module = RepaModule(
                module, hidden_shape, out_size
            )
            self.model.set_submodule(module_name, patched_module)

    def get_base_model(
            self
    ):
        """ Returns back the original model, without the REPA layers

        """
        for idx, module_name in enumerate(self.repa_config.target_modules):
            module = self.model.get_submodule(module_name)
            self.model.set_submodule(module_name, module.base_module)
        return self.model

    def forward(
            self,
            *args,
            **kwargs
    ):
        return self.model(
            *args,
            **kwargs
        )
    
    @torch.no_grad()
    def calculate_loss(
            self,
            features):
        
        loss = torch.tensor([0], device=self.device, dtype=self.dtype)
        for idx, module_name in enumerate(self.repa_config.target_modules):
            module = self.model.get_submodule(module_name)
            zs = module.zs

            # calculate here....
            zs = zs.mean(dim=1)
            loss = loss + torch.mean((zs - features)**2)

            # reset zs
            module.zs = None
        
        return loss / len(self.repa_config.target_modules)
    
    def save_pretrained(self, save_directory):
        """Saves the base model and the Repa modules."""
        device = self.device
        dtype = self.dtype
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 1. Save the base model using its native save_pretrained
        # To do this correctly, we must unpatch the model first.
        original_model = self.get_base_model()
        original_model.save_pretrained(save_directory)
        # Patch the model again immediately after saving
        self._patch_base_model()
        
        # 2. Save the Repa config
        config_path = os.path.join(save_directory, "repa_config.json")
        config_dict = self.repa_config.__dict__
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        
        # 3. Save the state dicts for the RepaMLP layers
        repa_state_dict = {}
        for module_name in self.repa_config.target_modules:
            # Note: The patched module is RepaModule, and the MLP is inside it
            mlp_module = self.model.get_submodule(module_name).proj_mlp
            repa_state_dict[module_name] = mlp_module.state_dict()
            
        repa_weights_path = os.path.join(save_directory, "repa_model_weights.bin")
        torch.save(repa_state_dict, repa_weights_path)
        self.to(device)
        self.to(dtype)
    
    @classmethod
    def from_pretrained(cls, base_model, pretrained_model_name_or_path, **kwargs):
        """Loads the base model and reconstructs the Repa modules."""
        # 2. Load the Repa config
        config_path = os.path.join(pretrained_model_name_or_path, "repa_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Repa config file not found at {config_path}")
            
        with open(config_path, "r") as f:
            repa_config_dict = json.load(f)
        repa_config = RepaConfig(**repa_config_dict)
        
        # 3. Instantiate the RepaModel, which automatically patches the base model
        repa_model = cls(base_model, repa_config)
        
        # 4. Load the weights for the RepaMLP layers
        repa_weights_path = os.path.join(pretrained_model_name_or_path, "repa_model_weights.bin")
        if os.path.exists(repa_weights_path):
            repa_state_dict = torch.load(repa_weights_path, map_location='cpu')
            
            for module_name, state_dict in repa_state_dict.items():
                mlp_module = repa_model.model.get_submodule(module_name).proj_mlp
                mlp_module.load_state_dict(state_dict)
        
        return repa_model
    

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)
        
