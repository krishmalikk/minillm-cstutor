"""
LoRA (Low-Rank Adaptation) module for CS Tutor LLM.
"""

from .lora import (
    LoRAConfig,
    LoRALinear,
    LoRAModel,
    apply_lora_to_model,
    merge_lora_weights,
    unmerge_lora_weights,
    save_lora_weights,
    load_lora_weights,
    get_lora_state_dict,
    set_lora_state_dict,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAModel",
    "apply_lora_to_model",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    "get_lora_state_dict",
    "set_lora_state_dict",
]

