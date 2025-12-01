"""
Data processing module for CS Tutor LLM.
"""

from .schema import (
    TaskType,
    Topic,
    Difficulty,
    InstructionExample,
    QuizQuestion,
    PracticeProblem,
    GradingExample,
    PROMPT_TEMPLATES,
)

from .dataset import (
    DataConfig,
    InstructionDataset,
    StreamingDataset,
    load_jsonl,
    save_jsonl,
    split_dataset,
    create_dataloaders,
    prepare_dataset,
    collate_fn,
)

__all__ = [
    # Schema
    "TaskType",
    "Topic",
    "Difficulty",
    "InstructionExample",
    "QuizQuestion",
    "PracticeProblem",
    "GradingExample",
    "PROMPT_TEMPLATES",
    # Dataset
    "DataConfig",
    "InstructionDataset",
    "StreamingDataset",
    "load_jsonl",
    "save_jsonl",
    "split_dataset",
    "create_dataloaders",
    "prepare_dataset",
    "collate_fn",
]

