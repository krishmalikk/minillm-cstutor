#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for CS Tutor LLM.

Fine-tune a pretrained model using LoRA (Low-Rank Adaptation)
for efficient parameter-efficient training.

Usage:
    python scripts/finetune_lora.py \
        --base-model outputs/final_model \
        --data-dir data/examples \
        --output-dir outputs/lora_finetuned
    
    # Custom LoRA rank
    python scripts/finetune_lora.py \
        --base-model outputs/final_model \
        --lora-rank 32 \
        --lora-alpha 64
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.optim import AdamW

from src.model import CSTutorForCausalLM, ModelConfig, CSTutorTokenizer
from src.data import DataConfig, load_jsonl, split_dataset, create_dataloaders
from src.lora import LoRAConfig, LoRAModel, save_lora_weights, merge_lora_weights
from src.training import TrainingConfig, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tune CS Tutor LLM")
    
    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to pretrained base model",
    )
    
    # LoRA config
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (r)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Modules to apply LoRA to",
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/examples",
        help="Directory with fine-tuning data",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    # Training
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/lora_finetuned",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (typically lower for fine-tuning)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    
    # Output options
    parser.add_argument(
        "--merge-weights",
        action="store_true",
        help="Merge LoRA weights into base model after training",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save the merged model (requires --merge-weights)",
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("LoRA Fine-tuning for CS Tutor LLM")
    print("="*60)
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load base model
    print(f"\n[1/5] Loading base model from {args.base_model}...")
    
    # Check if it's a directory or model config
    base_path = Path(args.base_model)
    
    if (base_path / "config.yaml").exists():
        config = ModelConfig.from_pretrained(str(base_path))
        model = CSTutorForCausalLM(config)
        model_weights = base_path / "model.pt"
        if model_weights.exists():
            model.load_state_dict(torch.load(model_weights, map_location="cpu"))
            print(f"  Loaded weights from {model_weights}")
    else:
        # Create a new model with default config
        from src.model import get_model_config
        config = get_model_config("125m")
        model = CSTutorForCausalLM(config)
        print("  Using fresh model (no pretrained weights found)")
    
    print(f"  Model: {config.name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer_path = base_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = CSTutorTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = CSTutorTokenizer(vocab_size=config.vocab_size)
    print(f"  Vocabulary size: {len(tokenizer)}")
    
    # Apply LoRA
    print("\n[3/5] Applying LoRA adapters...")
    lora_config = LoRAConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    
    lora_model = LoRAModel(model, lora_config)
    
    # Load data
    print(f"\n[4/5] Loading data from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    
    all_examples = []
    for jsonl_file in data_dir.glob("**/*.jsonl"):
        examples = load_jsonl(str(jsonl_file))
        all_examples.extend(examples)
        print(f"  Loaded {len(examples)} from {jsonl_file.name}")
    
    if not all_examples:
        print("ERROR: No training data found!")
        sys.exit(1)
    
    data_config = DataConfig(
        max_seq_length=args.max_seq_length,
        train_split=0.9,
        eval_split=0.1,
        seed=args.seed,
    )
    
    train_examples, eval_examples, _ = split_dataset(all_examples, data_config)
    print(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")
    
    train_loader, eval_loader = create_dataloaders(
        train_examples,
        eval_examples,
        tokenizer,
        data_config,
        batch_size=args.batch_size,
    )
    
    # Setup training
    print("\n[5/5] Starting LoRA fine-tuning...")
    
    training_config = TrainingConfig(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        bf16=args.bf16,
        logging_steps=10,
        eval_steps=200,
        save_steps=500,
        seed=args.seed,
    )
    
    trainer = Trainer(
        model=lora_model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=training_config,
    )
    
    # Train
    train_losses, eval_losses = trainer.train()
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights
    lora_weights_path = output_dir / "lora_weights.pt"
    lora_model.save_lora(str(lora_weights_path))
    
    # Save LoRA config
    import json
    lora_config_path = output_dir / "lora_config.json"
    with open(lora_config_path, 'w') as f:
        json.dump({
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules,
        }, f, indent=2)
    
    # Optionally merge and save
    if args.merge_weights:
        print("\nMerging LoRA weights into base model...")
        lora_model.merge()
        
        if args.save_merged:
            merged_path = output_dir / "merged_model"
            lora_model.get_base_model().save_pretrained(str(merged_path))
            tokenizer.save(str(merged_path / "tokenizer"))
            print(f"Merged model saved to {merged_path}")
    
    print("\n" + "="*60)
    print("LoRA Fine-tuning Complete!")
    print("="*60)
    print(f"LoRA weights: {lora_weights_path}")
    print(f"LoRA config: {lora_config_path}")
    if args.merge_weights and args.save_merged:
        print(f"Merged model: {output_dir / 'merged_model'}")


if __name__ == "__main__":
    main()

