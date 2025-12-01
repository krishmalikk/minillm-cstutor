"""
CS Tutor LLM - A Local, Offline Computer Science Tutor.

This package provides a complete system for:
- Training GPT-style language models for CS tutoring
- Instruction-tuning with LoRA
- Quantization for efficient inference
- CLI interface for interactive tutoring

Modules:
- model: GPT-style transformer architecture
- data: Dataset processing and loading
- training: Training pipeline
- lora: LoRA fine-tuning
- quantization: Model compression
- inference: High-level tutoring API
- cli: Command-line interface
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .model import CSTutorLLM, CSTutorForCausalLM, ModelConfig, CSTutorTokenizer
from .inference import CSTutorEngine

__all__ = [
    "CSTutorLLM",
    "CSTutorForCausalLM",
    "ModelConfig",
    "CSTutorTokenizer",
    "CSTutorEngine",
    "__version__",
]

