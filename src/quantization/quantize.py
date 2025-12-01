"""
Quantization Pipeline for CS Tutor LLM.

Supports:
- 8-bit quantization (INT8)
- 4-bit quantization (NF4/FP4)
- GGUF export for llama.cpp
"""

import os
import struct
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import numpy as np


class QuantizationType(Enum):
    """Supported quantization types."""
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # 4-bit NormalFloat
    FP4 = "fp4"  # 4-bit FloatingPoint


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Quantization type
    quant_type: QuantizationType = QuantizationType.INT8
    
    # Modules to skip (keep in original precision)
    skip_modules: List[str] = None
    
    # Group size for group-wise quantization
    group_size: int = 128
    
    # Use double quantization (quantize the scales)
    double_quant: bool = True
    
    # Compute dtype for operations
    compute_dtype: torch.dtype = torch.bfloat16
    
    def __post_init__(self):
        if self.skip_modules is None:
            self.skip_modules = ["lm_head", "embed_tokens"]


class Int8Quantizer:
    """
    INT8 dynamic quantization.
    
    Quantizes weights to 8-bit integers with per-channel scaling.
    """
    
    @staticmethod
    def quantize_tensor(
        tensor: torch.Tensor,
        per_channel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to INT8.
        
        Args:
            tensor: Input tensor (float)
            per_channel: Use per-channel scaling
        
        Returns:
            Tuple of (quantized_tensor, scale)
        """
        if per_channel and tensor.dim() >= 2:
            # Per-output-channel quantization
            dim = 0
            max_vals = tensor.abs().amax(dim=tuple(range(1, tensor.dim())), keepdim=True)
        else:
            max_vals = tensor.abs().max()
        
        scale = max_vals / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        
        return quantized, scale.squeeze()
    
    @staticmethod
    def dequantize_tensor(
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize INT8 tensor back to float."""
        if scale.dim() == 0:
            return quantized.float() * scale
        else:
            # Reshape scale for broadcasting
            shape = [1] * quantized.dim()
            shape[0] = -1
            scale = scale.view(shape)
            return quantized.float() * scale


class Int4Quantizer:
    """
    INT4 quantization with group-wise scaling.
    
    Packs two 4-bit values into one byte for memory efficiency.
    """
    
    def __init__(self, group_size: int = 128, use_nf4: bool = True):
        self.group_size = group_size
        self.use_nf4 = use_nf4
        
        # NF4 quantization levels (optimized for normal distribution)
        if use_nf4:
            self.nf4_values = torch.tensor([
                -1.0, -0.6961928009986877, -0.5250730514526367,
                -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725,
                0.24611aborar6085243225, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0,
            ])
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize tensor to 4-bit.
        
        Returns:
            (quantized, scales, zeros) - zeros is None for NF4
        """
        original_shape = tensor.shape
        tensor = tensor.view(-1)
        
        # Pad to multiple of group_size
        pad_len = (self.group_size - len(tensor) % self.group_size) % self.group_size
        if pad_len > 0:
            tensor = torch.cat([tensor, torch.zeros(pad_len, dtype=tensor.dtype)])
        
        # Reshape into groups
        tensor = tensor.view(-1, self.group_size)
        
        if self.use_nf4:
            # NF4 quantization
            scales = tensor.abs().amax(dim=1, keepdim=True)
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            
            # Normalize to [-1, 1]
            normalized = tensor / scales
            
            # Find nearest NF4 value
            nf4_vals = self.nf4_values.to(tensor.device)
            distances = (normalized.unsqueeze(-1) - nf4_vals).abs()
            quantized = distances.argmin(dim=-1).to(torch.uint8)
            
            return quantized.view(-1)[:len(tensor.view(-1)) - pad_len], scales.squeeze(), None
        else:
            # Standard INT4 with zero-point
            min_vals = tensor.amin(dim=1, keepdim=True)
            max_vals = tensor.amax(dim=1, keepdim=True)
            
            scales = (max_vals - min_vals) / 15.0
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            zeros = min_vals
            
            quantized = torch.round((tensor - zeros) / scales).clamp(0, 15).to(torch.uint8)
            
            return quantized.view(-1)[:len(tensor.view(-1)) - pad_len], scales.squeeze(), zeros.squeeze()
    
    @staticmethod
    def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
        """Pack two INT4 values into one byte."""
        assert tensor.dtype == torch.uint8
        assert len(tensor) % 2 == 0
        
        tensor = tensor.view(-1, 2)
        packed = (tensor[:, 0] & 0x0F) | ((tensor[:, 1] & 0x0F) << 4)
        return packed
    
    @staticmethod
    def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
        """Unpack INT4 values from packed bytes."""
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        return torch.stack([low, high], dim=1).view(-1)


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer.
    
    Stores weights in quantized format and dequantizes on-the-fly
    for computation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: QuantizationType = QuantizationType.INT8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        
        # Quantized weights (stored as buffer, not parameter)
        self.register_buffer("weight_quantized", None)
        self.register_buffer("weight_scale", None)
        self.register_buffer("weight_zeros", None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quant_type: QuantizationType = QuantizationType.INT8,
    ) -> "QuantizedLinear":
        """Create quantized linear from regular linear."""
        quantized = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            quant_type=quant_type,
        )
        
        # Quantize weights
        if quant_type == QuantizationType.INT8:
            q_weight, scale = Int8Quantizer.quantize_tensor(linear.weight.data)
            quantized.weight_quantized = q_weight
            quantized.weight_scale = scale
        elif quant_type in [QuantizationType.INT4, QuantizationType.NF4]:
            quantizer = Int4Quantizer(use_nf4=(quant_type == QuantizationType.NF4))
            q_weight, scale, zeros = quantizer.quantize_tensor(linear.weight.data)
            quantized.weight_quantized = Int4Quantizer.pack_int4(q_weight)
            quantized.weight_scale = scale
            if zeros is not None:
                quantized.weight_zeros = zeros
        
        # Copy bias
        if linear.bias is not None:
            quantized.bias.data = linear.bias.data.clone()
        
        return quantized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with on-the-fly dequantization."""
        # Dequantize weights
        if self.quant_type == QuantizationType.INT8:
            weight = Int8Quantizer.dequantize_tensor(
                self.weight_quantized,
                self.weight_scale,
            )
        else:
            # For INT4/NF4, implementation would be more complex
            # This is a simplified version
            unpacked = Int4Quantizer.unpack_int4(self.weight_quantized)
            weight = unpacked.float().view(self.out_features, self.in_features)
            weight = weight * self.weight_scale.view(-1, 1)
        
        weight = weight.to(x.dtype)
        return nn.functional.linear(x, weight, self.bias)


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """
    Quantize all linear layers in a model.
    
    Args:
        model: Model to quantize
        config: Quantization configuration
    
    Returns:
        Quantized model
    """
    skip_modules = set(config.skip_modules)
    
    def _quantize_module(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # Check if should skip
                should_skip = any(skip in full_name for skip in skip_modules)
                
                if not should_skip:
                    quantized = QuantizedLinear.from_linear(child, config.quant_type)
                    setattr(module, name, quantized)
                    print(f"  Quantized {full_name}")
            else:
                _quantize_module(child, full_name)
    
    print(f"Quantizing model to {config.quant_type.value}...")
    _quantize_module(model)
    
    # Calculate size reduction
    original_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    quantized_size = sum(
        b.numel() * b.element_size() 
        for b in model.buffers() 
        if b is not None
    ) + sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    
    print(f"\nQuantization complete:")
    print(f"  Original size: {original_size / 1024**2:.2f} MB")
    print(f"  Quantized size: {quantized_size / 1024**2:.2f} MB")
    print(f"  Compression ratio: {original_size / quantized_size:.2f}x")
    
    return model


class GGUFWriter:
    """
    GGUF format writer for llama.cpp compatibility.
    
    GGUF (GPT-Generated Unified Format) is the standard format
    for running models with llama.cpp.
    """
    
    # GGUF constants
    GGUF_MAGIC = 0x46554747  # "GGUF"
    GGUF_VERSION = 3
    
    # Value types
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Tuple[str, torch.Tensor, str]] = []
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value
    
    def add_tensor(self, name: str, tensor: torch.Tensor, quant_type: str = "f32"):
        """Add tensor to be written."""
        self.tensors.append((name, tensor, quant_type))
    
    def write(self):
        """Write GGUF file."""
        with open(self.output_path, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', self.GGUF_MAGIC))
            f.write(struct.pack('<I', self.GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))
            
            # Write metadata
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_value(f, value)
            
            # Write tensor info
            tensor_data_offset = f.tell()
            for name, tensor, quant_type in self.tensors:
                self._write_string(f, name)
                f.write(struct.pack('<I', len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', self._get_ggml_type(quant_type)))
                f.write(struct.pack('<Q', tensor_data_offset))
                tensor_data_offset += tensor.numel() * tensor.element_size()
            
            # Align to 32 bytes
            current = f.tell()
            padding = (32 - current % 32) % 32
            f.write(b'\x00' * padding)
            
            # Write tensor data
            for name, tensor, quant_type in self.tensors:
                tensor_np = tensor.cpu().numpy()
                f.write(tensor_np.tobytes())
        
        print(f"GGUF file written to {self.output_path}")
    
    def _write_string(self, f, s: str):
        """Write a string."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f, value):
        """Write a typed value."""
        if isinstance(value, str):
            f.write(struct.pack('<I', self.GGUF_TYPE_STRING))
            self._write_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', self.GGUF_TYPE_INT32))
            f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', self.GGUF_TYPE_FLOAT32))
            f.write(struct.pack('<f', value))
    
    def _get_ggml_type(self, quant_type: str) -> int:
        """Get GGML type code."""
        type_map = {
            "f32": 0,
            "f16": 1,
            "q4_0": 2,
            "q4_1": 3,
            "q5_0": 6,
            "q5_1": 7,
            "q8_0": 8,
        }
        return type_map.get(quant_type, 0)


def export_to_gguf(
    model: nn.Module,
    tokenizer,
    output_path: str,
    quant_type: str = "q4_0",
    model_name: str = "cs-tutor",
):
    """
    Export model to GGUF format.
    
    Args:
        model: The model to export
        tokenizer: Tokenizer
        output_path: Output GGUF file path
        quant_type: Quantization type (q4_0, q8_0, f16, f32)
        model_name: Model name for metadata
    """
    writer = GGUFWriter(output_path)
    
    # Add metadata
    writer.add_metadata("general.architecture", "llama")
    writer.add_metadata("general.name", model_name)
    writer.add_metadata("llama.vocab_size", len(tokenizer))
    
    if hasattr(model, 'config'):
        config = model.config
        writer.add_metadata("llama.context_length", config.max_seq_length)
        writer.add_metadata("llama.embedding_length", config.hidden_size)
        writer.add_metadata("llama.block_count", config.num_hidden_layers)
        writer.add_metadata("llama.attention.head_count", config.num_attention_heads)
        writer.add_metadata("llama.rope.freq_base", config.rope_theta)
    
    # Add tensors
    for name, param in model.named_parameters():
        # Convert name to GGUF format
        gguf_name = name.replace(".", "/")
        writer.add_tensor(gguf_name, param.data, quant_type)
    
    writer.write()
    
    print(f"\nExported to GGUF: {output_path}")
    print(f"  Quantization: {quant_type}")
    print(f"  File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")

