"""
Training Pipeline for CS Tutor LLM.

Full-featured trainer with:
- AdamW optimizer
- Mixed precision (fp16/bf16)
- Gradient checkpointing
- Cosine LR schedule with warmup
- Gradient accumulation
- Checkpointing and resumption
- Logging and metrics
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Output
    output_dir: str = "outputs/checkpoints"
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs
    
    # Batch size
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    
    # Learning rate
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    
    # Optimizer
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True
    
    # Logging & Evaluation
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4


class CosineSchedulerWithWarmup:
    """
    Cosine learning rate scheduler with linear warmup.
    
    LR starts at 0, linearly increases during warmup,
    then decreases following a cosine curve to min_lr.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def get_lr(self, step: int) -> List[float]:
        """Calculate learning rate for given step."""
        if step < self.num_warmup_steps:
            # Linear warmup
            warmup_ratio = step / max(1, self.num_warmup_steps)
            return [lr * warmup_ratio for lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.num_warmup_steps) / max(
                1, self.num_training_steps - self.num_warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
            return [lr * decayed for lr in self.base_lrs]
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lrs = self.get_lr(self.current_step)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
    
    def state_dict(self) -> Dict:
        return {
            "current_step": self.current_step,
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.current_step = state_dict["current_step"]
        self.num_warmup_steps = state_dict["num_warmup_steps"]
        self.num_training_steps = state_dict["num_training_steps"]
        self.min_lr_ratio = state_dict["min_lr_ratio"]
        self.base_lrs = state_dict["base_lrs"]


class Trainer:
    """
    Full-featured trainer for CS Tutor LLM.
    
    Features:
    - Mixed precision training (fp16/bf16)
    - Gradient accumulation
    - Gradient checkpointing
    - Cosine LR schedule with warmup
    - Checkpointing and resumption
    - Evaluation during training
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup gradient checkpointing
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Calculate training steps
        self.num_training_steps = self._calculate_training_steps()
        self.num_warmup_steps = self._calculate_warmup_steps()
        
        # Setup scheduler
        self.scheduler = CosineSchedulerWithWarmup(
            self.optimizer,
            self.num_warmup_steps,
            self.num_training_steps,
            min_lr_ratio=self.config.min_learning_rate / self.config.learning_rate,
        )
        
        # Setup mixed precision
        self.scaler = None
        self.autocast_dtype = None
        self._setup_mixed_precision()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Metrics
        self.train_losses = []
        self.eval_losses = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'layernorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def _calculate_training_steps(self) -> int:
        """Calculate total number of training steps."""
        if self.config.max_steps > 0:
            return self.config.max_steps
        
        num_batches = len(self.train_dataloader)
        steps_per_epoch = num_batches // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_train_epochs
    
    def _calculate_warmup_steps(self) -> int:
        """Calculate number of warmup steps."""
        if self.config.warmup_steps > 0:
            return self.config.warmup_steps
        return int(self.num_training_steps * self.config.warmup_ratio)
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.config.bf16 and self.device.type == "cuda":
            self.autocast_dtype = torch.bfloat16
        elif self.config.fp16 and self.device.type == "cuda":
            self.autocast_dtype = torch.float16
            self.scaler = GradScaler()
        else:
            self.autocast_dtype = None
    
    def _get_autocast_context(self):
        """Get autocast context manager."""
        if self.autocast_dtype is not None and self.device.type == "cuda":
            return autocast(dtype=self.autocast_dtype)
        return nullcontext()
    
    def train(self):
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Training steps: {self.num_training_steps}")
        print(f"Warmup steps: {self.num_warmup_steps}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        print(f"{'='*60}\n")
        
        self.model.train()
        
        start_time = time.time()
        accumulated_loss = 0.0
        num_accumulated = 0
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")
            print("-" * 40)
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with self._get_autocast_context():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                num_accumulated += 1
                
                # Gradient accumulation
                if num_accumulated >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Log metrics
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss * self.config.gradient_accumulation_steps
                        self.train_losses.append(avg_loss)
                        current_lr = self.optimizer.param_groups[0]['lr']
                        elapsed = time.time() - start_time
                        steps_per_sec = self.global_step / elapsed
                        
                        print(
                            f"Step {self.global_step}/{self.num_training_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Speed: {steps_per_sec:.2f} steps/s"
                        )
                    
                    # Reset accumulation
                    accumulated_loss = 0.0
                    num_accumulated = 0
                    
                    # Evaluation
                    if (self.eval_dataloader is not None and 
                        self.global_step % self.config.eval_steps == 0):
                        eval_loss = self.evaluate()
                        self.eval_losses.append(eval_loss)
                        print(f"Eval Loss: {eval_loss:.4f}")
                        
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint("best")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step-{self.global_step}")
                    
                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} completed in {epoch_time/60:.2f} minutes")
            
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
        
        # Final evaluation
        if self.eval_dataloader is not None:
            final_loss = self.evaluate()
            print(f"\nFinal Eval Loss: {final_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint("final")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best eval loss: {self.best_eval_loss:.4f}")
        
        return self.train_losses, self.eval_losses
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with self._get_autocast_context():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler
        scheduler_path = checkpoint_dir / "scheduler.pt"
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "config": asdict(self.config),
        }
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save model config if available
        if hasattr(self.model, 'config'):
            self.model.config.save(str(checkpoint_dir / "config.yaml"))
        
        print(f"Checkpoint saved to {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return
        
        output_dir = Path(self.config.output_dir)
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
            key=lambda x: int(x.name.split("-")[1]),
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            checkpoint_to_remove = checkpoints.pop(0)
            import shutil
            shutil.rmtree(checkpoint_to_remove)
            print(f"Removed old checkpoint: {checkpoint_to_remove}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        model_path = checkpoint_dir / "model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load scheduler
        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        
        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_eval_loss = state["best_eval_loss"]
            self.train_losses = state["train_losses"]
            self.eval_losses = state["eval_losses"]
        
        print(f"Loaded checkpoint from {checkpoint_dir}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")

