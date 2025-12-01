#!/usr/bin/env python3
"""
Quantization Script for CS Tutor LLM.

Quantize trained models for efficient inference.

Usage:
    # INT8 quantization
    python scripts/quantize.py \
        --model-path outputs/final_model \
        --output-dir outputs/quantized \
        --quant-type int8
    
    # 4-bit quantization
    python scripts/quantize.py \
        --model-path outputs/final_model \
        --output-dir outputs/quantized \
        --quant-type nf4
    
    # Export to GGUF for llama.cpp
    python scripts/quantize.py \
        --model-path outputs/final_model \
        --output-dir outputs/gguf \
        --export-gguf \
        --gguf-quant q4_0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.model import CSTutorLLM, ModelConfig, CSTutorTokenizer
from src.quantization import (
    QuantizationType,
    QuantizationConfig,
    quantize_model,
    export_to_gguf,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize CS Tutor LLM")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="int8",
        choices=["int8", "int4", "nf4", "fp4"],
        help="Quantization type",
    )
    parser.add_argument(
        "--skip-modules",
        type=str,
        nargs="+",
        default=["lm_head", "embed_tokens"],
        help="Modules to skip quantization",
    )
    
    # GGUF export
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export to GGUF format",
    )
    parser.add_argument(
        "--gguf-quant",
        type=str,
        default="q4_0",
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="GGUF quantization type",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("CS Tutor LLM Quantization")
    print("="*60)
    
    # Load model
    print(f"\n[1/3] Loading model from {args.model_path}...")
    
    model_path = Path(args.model_path)
    
    if (model_path / "config.yaml").exists():
        config = ModelConfig.from_pretrained(str(model_path))
        model = CSTutorLLM(config)
        
        weights_path = model_path / "model.pt"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"  Loaded weights from {weights_path}")
    else:
        print("ERROR: Model config not found!")
        sys.exit(1)
    
    # Load tokenizer
    tokenizer_path = model_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = CSTutorTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = CSTutorTokenizer(vocab_size=config.vocab_size)
    
    # Calculate original size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Original model size: {original_size / 1024**2:.2f} MB")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.export_gguf:
        # Export to GGUF format
        print(f"\n[2/3] Exporting to GGUF format ({args.gguf_quant})...")
        
        gguf_path = output_dir / f"cs-tutor-{args.gguf_quant}.gguf"
        export_to_gguf(
            model,
            tokenizer,
            str(gguf_path),
            quant_type=args.gguf_quant,
            model_name="cs-tutor",
        )
        
        print(f"\n[3/3] GGUF export complete!")
        print(f"  Output: {gguf_path}")
        print(f"  Size: {os.path.getsize(gguf_path) / 1024**2:.2f} MB")
        
    else:
        # PyTorch quantization
        print(f"\n[2/3] Quantizing model ({args.quant_type})...")
        
        quant_type_map = {
            "int8": QuantizationType.INT8,
            "int4": QuantizationType.INT4,
            "nf4": QuantizationType.NF4,
            "fp4": QuantizationType.FP4,
        }
        
        quant_config = QuantizationConfig(
            quant_type=quant_type_map[args.quant_type],
            skip_modules=args.skip_modules,
        )
        
        quantized_model = quantize_model(model, quant_config)
        
        # Save quantized model
        print(f"\n[3/3] Saving quantized model...")
        
        model_output = output_dir / "model.pt"
        torch.save(quantized_model.state_dict(), model_output)
        
        # Save config
        config.save(str(output_dir / "config.yaml"))
        
        # Save tokenizer
        tokenizer.save(str(output_dir / "tokenizer"))
        
        # Save quantization info
        import json
        quant_info = {
            "quant_type": args.quant_type,
            "skip_modules": args.skip_modules,
            "original_size_mb": original_size / 1024**2,
            "quantized_size_mb": os.path.getsize(model_output) / 1024**2,
        }
        with open(output_dir / "quantization_info.json", 'w') as f:
            json.dump(quant_info, f, indent=2)
        
        print(f"\nQuantization complete!")
        print(f"  Output directory: {output_dir}")
        print(f"  Model: {model_output}")
        print(f"  Size: {os.path.getsize(model_output) / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()

