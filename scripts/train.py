#!/usr/bin/env python3
"""
Training Script for CS Tutor LLM.

Usage:
    python scripts/train.py --model-size 125m --data-dir data/examples --output-dir outputs
    
    # Resume from checkpoint
    python scripts/train.py --resume outputs/checkpoints/step-1000
    
    # Custom config
    python scripts/train.py --config configs/training_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.model import CSTutorForCausalLM, ModelConfig, get_model_config, CSTutorTokenizer
from src.data import (
    DataConfig,
    load_jsonl,
    split_dataset,
    create_dataloaders,
)
from src.training.trainer import Trainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CS Tutor LLM")
    
    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        default="125m",
        choices=["125m", "300m", "1b"],
        help="Model size configuration",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to custom model config YAML",
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/examples",
        help="Directory containing training JSONL files",
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
        default="outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    
    # Mixed precision
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 mixed precision (default)",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16",
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    
    # Logging
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_config(args) -> TrainingConfig:
    """Load training configuration from args or YAML."""
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        training_dict = config_dict.get('training', {})
        return TrainingConfig(**training_dict)
    
    # Build from command line args
    bf16 = args.bf16 and not args.no_bf16
    
    return TrainingConfig(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=bf16,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
    )


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("CS Tutor LLM Training")
    print("="*60)
    
    # Set seed
    set_seed(args.seed)
    
    # Load model config
    print(f"\n[1/5] Loading model configuration ({args.model_size})...")
    if args.model_config:
        model_config = ModelConfig.from_yaml(args.model_config)
    else:
        model_config = get_model_config(args.model_size)
    
    print(f"  Model: {model_config.name}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_hidden_layers}")
    print(f"  Attention heads: {model_config.num_attention_heads}")
    print(f"  Estimated parameters: {model_config.num_parameters:,}")
    
    # Initialize tokenizer
    print("\n[2/5] Initializing tokenizer...")
    tokenizer = CSTutorTokenizer(vocab_size=model_config.vocab_size)
    
    # For a real training run, you would train the tokenizer on your corpus:
    # tokenizer.train(your_texts, vocab_size=32000)
    
    # Or load a pretrained tokenizer:
    # tokenizer = CSTutorTokenizer.from_pretrained("path/to/tokenizer")
    
    print(f"  Vocabulary size: {model_config.vocab_size}")
    
    # Load data
    print(f"\n[3/5] Loading data from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    
    all_examples = []
    for jsonl_file in data_dir.glob("**/*.jsonl"):
        examples = load_jsonl(str(jsonl_file))
        all_examples.extend(examples)
        print(f"  Loaded {len(examples)} examples from {jsonl_file.name}")
    
    if not all_examples:
        print("ERROR: No training examples found!")
        print(f"Please add JSONL files to {args.data_dir}")
        sys.exit(1)
    
    print(f"  Total examples: {len(all_examples)}")
    
    # Split data
    data_config = DataConfig(
        max_seq_length=args.max_seq_length,
        train_split=0.9,
        eval_split=0.1,
        seed=args.seed,
    )
    
    train_examples, eval_examples, _ = split_dataset(all_examples, data_config)
    print(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")
    
    # Create dataloaders
    train_loader, eval_loader = create_dataloaders(
        train_examples,
        eval_examples,
        tokenizer,
        data_config,
        batch_size=args.batch_size,
    )
    
    # Initialize model
    print("\n[4/5] Initializing model...")
    model = CSTutorForCausalLM(model_config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size: {num_params * 4 / 1024**3:.2f} GB (fp32)")
    
    # Load training config
    training_config = load_training_config(args)
    
    # Initialize trainer
    print("\n[5/5] Starting training...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=training_config,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    train_losses, eval_losses = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save(os.path.join(final_model_path, "tokenizer"))
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {final_model_path}")
    
    # Save training metrics
    metrics = {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "best_eval_loss": trainer.best_eval_loss,
    }
    
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

