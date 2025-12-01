"""
CS Tutor LLM Model Package.

A GPT-style decoder-only transformer for teaching CS concepts.
"""

from .config import ModelConfig, get_model_config, MODEL_CONFIGS
from .model import CSTutorLLM, CSTutorForCausalLM, CausalLMOutput
from .tokenizer import CSTutorTokenizer, load_llama_tokenizer
from .layers import (
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    Attention,
    TransformerBlock,
)

__all__ = [
    # Config
    "ModelConfig",
    "get_model_config",
    "MODEL_CONFIGS",
    # Models
    "CSTutorLLM",
    "CSTutorForCausalLM",
    "CausalLMOutput",
    # Tokenizer
    "CSTutorTokenizer",
    "load_llama_tokenizer",
    # Layers
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
    "Attention",
    "TransformerBlock",
]

