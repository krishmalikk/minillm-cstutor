"""
Dataset utilities for CS Tutor LLM.

Handles loading, processing, and preparing instruction-tuning datasets.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Callable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .schema import InstructionExample


@dataclass
class DataConfig:
    """Configuration for dataset processing."""
    max_seq_length: int = 2048
    train_split: float = 0.9
    eval_split: float = 0.05
    test_split: float = 0.05
    shuffle: bool = True
    seed: int = 42
    
    # Prompt formatting
    prompt_template: str = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
    
    # For cases with no input
    prompt_template_no_input: str = """### Instruction:
{instruction}

### Response:
{output}"""


class InstructionDataset(Dataset):
    """
    PyTorch Dataset for instruction-tuning examples.
    
    Handles tokenization and formatting of examples
    into the training format.
    """
    
    def __init__(
        self,
        examples: List[InstructionExample],
        tokenizer,
        config: DataConfig,
        is_training: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            examples: List of InstructionExample objects
            tokenizer: Tokenizer with encode/decode methods
            config: DataConfig with processing settings
            is_training: Whether this is for training (includes labels)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.examples[idx]
        
        # Format the prompt
        if example.input.strip():
            prompt = self.config.prompt_template.format(
                instruction=example.instruction,
                input=example.input,
                output=example.output,
            )
        else:
            prompt = self.config.prompt_template_no_input.format(
                instruction=example.instruction,
                output=example.output,
            )
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            add_bos=True,
            add_eos=True,
            max_length=self.config.max_seq_length,
            padding=False,
        )
        
        input_ids = encoded["input_ids"][0] if isinstance(encoded["input_ids"][0], list) else encoded["input_ids"]
        
        # Truncate if necessary
        if len(input_ids) > self.config.max_seq_length:
            input_ids = input_ids[:self.config.max_seq_length]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # For training, labels are same as input_ids (shifted internally during loss computation)
        if self.is_training:
            labels = input_ids.copy()
        else:
            labels = [-100] * len(input_ids)  # Ignore in loss
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads sequences to the same length within a batch.
    """
    # Find max length in batch
    max_length = max(item["input_ids"].size(0) for item in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        seq_len = item["input_ids"].size(0)
        padding_length = max_length - seq_len
        
        # Pad input_ids
        padded_input = torch.cat([
            item["input_ids"],
            torch.full((padding_length,), pad_token_id, dtype=torch.long),
        ])
        input_ids.append(padded_input)
        
        # Pad attention_mask (0 for padding)
        padded_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(padding_length, dtype=torch.long),
        ])
        attention_mask.append(padded_mask)
        
        # Pad labels (-100 to ignore in loss)
        padded_labels = torch.cat([
            item["labels"],
            torch.full((padding_length,), -100, dtype=torch.long),
        ])
        labels.append(padded_labels)
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def load_jsonl(file_path: str) -> List[InstructionExample]:
    """Load examples from a JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                examples.append(InstructionExample.from_dict(data))
    return examples


def save_jsonl(examples: List[InstructionExample], file_path: str):
    """Save examples to a JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(example.to_jsonl() + '\n')


def split_dataset(
    examples: List[InstructionExample],
    config: DataConfig,
) -> tuple:
    """
    Split dataset into train/eval/test sets.
    
    Args:
        examples: List of all examples
        config: DataConfig with split ratios
    
    Returns:
        Tuple of (train_examples, eval_examples, test_examples)
    """
    if config.shuffle:
        random.seed(config.seed)
        examples = examples.copy()
        random.shuffle(examples)
    
    n = len(examples)
    train_end = int(n * config.train_split)
    eval_end = train_end + int(n * config.eval_split)
    
    train = examples[:train_end]
    eval_set = examples[train_end:eval_end]
    test = examples[eval_end:]
    
    return train, eval_set, test


def create_dataloaders(
    train_examples: List[InstructionExample],
    eval_examples: List[InstructionExample],
    tokenizer,
    config: DataConfig,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple:
    """
    Create DataLoaders for training and evaluation.
    
    Args:
        train_examples: Training examples
        eval_examples: Evaluation examples
        tokenizer: Tokenizer instance
        config: DataConfig
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    train_dataset = InstructionDataset(
        train_examples, tokenizer, config, is_training=True
    )
    eval_dataset = InstructionDataset(
        eval_examples, tokenizer, config, is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    return train_loader, eval_loader


def prepare_dataset(
    data_dir: str,
    output_dir: str,
    tokenizer,
    config: Optional[DataConfig] = None,
) -> Dict[str, str]:
    """
    Prepare dataset from raw JSONL files.
    
    Loads all JSONL files from data_dir, splits them,
    and saves to output_dir.
    
    Args:
        data_dir: Directory with source JSONL files
        output_dir: Directory to save processed files
        tokenizer: Tokenizer for validation
        config: DataConfig (uses defaults if None)
    
    Returns:
        Dict with paths to train/eval/test files
    """
    config = config or DataConfig()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all examples
    all_examples = []
    for jsonl_file in data_dir.glob("**/*.jsonl"):
        print(f"Loading {jsonl_file}...")
        examples = load_jsonl(str(jsonl_file))
        all_examples.extend(examples)
        print(f"  Loaded {len(examples)} examples")
    
    print(f"\nTotal examples: {len(all_examples)}")
    
    # Split dataset
    train, eval_set, test = split_dataset(all_examples, config)
    
    print(f"Train: {len(train)}, Eval: {len(eval_set)}, Test: {len(test)}")
    
    # Save splits
    paths = {
        "train": str(output_dir / "train.jsonl"),
        "eval": str(output_dir / "eval.jsonl"),
        "test": str(output_dir / "test.jsonl"),
    }
    
    save_jsonl(train, paths["train"])
    save_jsonl(eval_set, paths["eval"])
    save_jsonl(test, paths["test"])
    
    print(f"\nSaved to {output_dir}")
    
    return paths


class StreamingDataset:
    """
    Memory-efficient dataset that streams from disk.
    
    Useful for very large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        config: DataConfig,
        buffer_size: int = 10000,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.config = config
        self.buffer_size = buffer_size
        
        # Count total lines for length
        with open(file_path, 'r') as f:
            self.total_lines = sum(1 for _ in f)
    
    def __len__(self) -> int:
        return self.total_lines
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream examples from file."""
        buffer = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    example = InstructionExample.from_dict(data)
                    buffer.append(example)
                    
                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)
                        for ex in buffer:
                            yield self._process_example(ex)
                        buffer = []
        
        # Process remaining
        if buffer:
            random.shuffle(buffer)
            for ex in buffer:
                yield self._process_example(ex)
    
    def _process_example(self, example: InstructionExample) -> Dict[str, torch.Tensor]:
        """Process a single example."""
        if example.input.strip():
            prompt = self.config.prompt_template.format(
                instruction=example.instruction,
                input=example.input,
                output=example.output,
            )
        else:
            prompt = self.config.prompt_template_no_input.format(
                instruction=example.instruction,
                output=example.output,
            )
        
        encoded = self.tokenizer(
            prompt,
            add_bos=True,
            add_eos=True,
            max_length=self.config.max_seq_length,
        )
        
        input_ids = encoded["input_ids"][0] if isinstance(encoded["input_ids"][0], list) else encoded["input_ids"]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }

