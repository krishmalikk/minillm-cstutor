"""
LoRA (Low-Rank Adaptation) implementation for CS Tutor LLM.

LoRA enables efficient fine-tuning by injecting trainable
low-rank matrices into transformer layers while keeping
the original weights frozen.

Paper: https://arxiv.org/abs/2106.09685
"""

import math
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    # LoRA rank (lower = fewer parameters, higher = more capacity)
    r: int = 16
    
    # LoRA alpha (scaling factor, typically 2*r)
    lora_alpha: int = 32
    
    # Dropout probability for LoRA layers
    lora_dropout: float = 0.05
    
    # Target modules to apply LoRA to
    target_modules: List[str] = None
    
    # Whether to merge weights for inference
    merge_weights: bool = False
    
    # Fan-in/fan-out initialization
    fan_in_fan_out: bool = False
    
    # Bias handling: "none", "all", "lora_only"
    bias: str = "none"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply to attention and MLP layers
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # MLP
            ]


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    During training:
        output = x @ W + x @ A @ B * (alpha/r)
    
    During inference (merged):
        output = x @ (W + A @ B * (alpha/r))
    
    Where:
        - W: Original frozen weights
        - A: Low-rank matrix (r x in_features)
        - B: Low-rank matrix (out_features x r)
        - alpha/r: Scaling factor
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        
        # LoRA layers
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
            
            # Initialize A with Kaiming, B with zeros (makes initial LoRA = 0)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        if self.r > 0 and not self.merged:
            # Compute original + LoRA
            result = self.linear(x)
            lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result + lora_out
        else:
            # Merged or no LoRA
            return self.linear(x)
    
    def merge(self):
        """Merge LoRA weights into the original linear layer."""
        if self.r > 0 and not self.merged:
            # W_new = W + B @ A * scaling
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from the original linear layer."""
        if self.r > 0 and self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ) -> "LoRALinear":
        """Create LoRALinear from an existing nn.Linear."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None,
        )
        
        # Copy original weights
        lora_linear.linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_linear.linear.bias.data = linear.bias.data.clone()
        
        return lora_linear


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model.
    
    Returns a state dict containing only lora_A and lora_B parameters.
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def set_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Load LoRA parameters into model."""
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name] = param
    model.load_state_dict(model_state, strict=False)


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """
    Apply LoRA adapters to a model.
    
    Replaces targeted linear layers with LoRALinear layers.
    
    Args:
        model: The model to modify
        config: LoRA configuration
    
    Returns:
        Modified model with LoRA adapters
    """
    target_modules = set(config.target_modules)
    
    def _replace_linear(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # Check if this module should have LoRA applied
                should_apply = any(target in name for target in target_modules)
                
                if should_apply:
                    lora_linear = LoRALinear.from_linear(
                        child,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                    )
                    setattr(module, name, lora_linear)
                    print(f"  Applied LoRA to {full_name}")
            else:
                _replace_linear(child, full_name)
    
    print(f"Applying LoRA (r={config.r}, alpha={config.lora_alpha})")
    _replace_linear(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nLoRA applied:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into the base model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    print("LoRA weights merged into base model")


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights from the base model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
    print("LoRA weights unmerged from base model")


def save_lora_weights(model: nn.Module, save_path: str):
    """Save only LoRA weights to a file."""
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, save_path)
    print(f"LoRA weights saved to {save_path}")
    print(f"  Number of tensors: {len(lora_state)}")
    print(f"  Total size: {sum(t.numel() * t.element_size() for t in lora_state.values()) / 1024**2:.2f} MB")


def load_lora_weights(model: nn.Module, load_path: str, strict: bool = True):
    """Load LoRA weights into model."""
    lora_state = torch.load(load_path, map_location="cpu")
    set_lora_state_dict(model, lora_state)
    print(f"LoRA weights loaded from {load_path}")


class LoRAModel(nn.Module):
    """
    Wrapper that adds LoRA to any model.
    
    Usage:
        base_model = CSTutorLLM(config)
        lora_model = LoRAModel(base_model, lora_config)
        
        # Train only LoRA parameters
        optimizer = Adam(lora_model.trainable_parameters(), lr=1e-4)
        
        # For inference, merge weights
        lora_model.merge()
    """
    
    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.config = config
        self.base_model = apply_lora_to_model(base_model, config)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def trainable_parameters(self):
        """Return only trainable (LoRA) parameters."""
        return (p for p in self.parameters() if p.requires_grad)
    
    def merge(self):
        """Merge LoRA weights for efficient inference."""
        merge_lora_weights(self.base_model)
    
    def unmerge(self):
        """Unmerge LoRA weights to resume training."""
        unmerge_lora_weights(self.base_model)
    
    def save_lora(self, save_path: str):
        """Save only LoRA weights."""
        save_lora_weights(self.base_model, save_path)
    
    def load_lora(self, load_path: str):
        """Load LoRA weights."""
        load_lora_weights(self.base_model, load_path)
    
    def get_base_model(self) -> nn.Module:
        """Get the underlying base model."""
        return self.base_model

