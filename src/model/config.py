"""
Model Configuration Dataclasses for CS Tutor LLM.

Provides type-safe configuration for different model sizes
and architectural choices.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the CS Tutor LLM model architecture."""
    
    # Model identification
    name: str = "cs-tutor-125m"
    architecture: str = "gpt-decoder"
    
    # Core dimensions
    vocab_size: int = 32000
    max_seq_length: int = 2048
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # For GQA, None = MHA
    head_dim: int = 64
    
    # Positional encoding
    position_embedding_type: str = "rotary"
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    # Normalization
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    
    # Regularization
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Activation
    hidden_activation: str = "silu"  # For SwiGLU
    
    # Initialization
    initializer_range: float = 0.02
    
    # Embeddings
    tie_word_embeddings: bool = True
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
            
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"
    
    @classmethod
    def from_yaml(cls, config_path: str, model_key: str = "model_125m") -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        
        if model_key not in configs:
            raise ValueError(f"Model key '{model_key}' not found in config file")
        
        return cls(**configs[model_key])
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load configuration from a pretrained model directory."""
        config_path = Path(model_path) / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    def save(self, save_path: str):
        """Save configuration to YAML file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
    
    @property
    def num_parameters(self) -> int:
        """Estimate total number of parameters."""
        # Embeddings
        embedding_params = self.vocab_size * self.hidden_size
        
        # Each transformer layer
        # Attention: Q, K, V, O projections
        qkv_params = self.hidden_size * self.head_dim * (
            self.num_attention_heads + 2 * self.num_key_value_heads
        )
        o_params = self.hidden_size * self.hidden_size
        
        # MLP (SwiGLU has gate, up, down projections)
        mlp_params = 3 * self.hidden_size * self.intermediate_size
        
        # Layer norms (2 per layer)
        norm_params = 2 * self.hidden_size
        
        layer_params = qkv_params + o_params + mlp_params + norm_params
        total_layer_params = layer_params * self.num_hidden_layers
        
        # Final layer norm
        final_norm_params = self.hidden_size
        
        # LM head (tied with embeddings if tie_word_embeddings=True)
        lm_head_params = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        
        total = embedding_params + total_layer_params + final_norm_params + lm_head_params
        return total


# Predefined configurations for different model sizes
MODEL_CONFIGS = {
    "125m": ModelConfig(
        name="cs-tutor-125m",
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
    "300m": ModelConfig(
        name="cs-tutor-300m",
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    "1b": ModelConfig(
        name="cs-tutor-1b",
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=32,
        max_seq_length=4096,
    ),
}


def get_model_config(size: str = "125m") -> ModelConfig:
    """Get a predefined model configuration by size."""
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[size]

