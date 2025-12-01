"""
CS Tutor LLM - Main Model Implementation.

A GPT-style decoder-only transformer optimized for
teaching computer science concepts.
"""

from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import (
    RMSNorm,
    RotaryPositionEmbedding,
    TransformerBlock,
)


class CSTutorLLM(nn.Module):
    """
    CS Tutor Large Language Model.
    
    A decoder-only transformer with:
    - Rotary Position Embeddings (RoPE)
    - RMSNorm for stability
    - SwiGLU activation in MLP
    - Optional Grouped Query Attention (GQA)
    - KV caching for efficient inference
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        
        # Rotary position embeddings
        self.rotary_emb = RotaryPositionEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_seq_length,
            base=config.rope_theta,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                hidden_dropout=config.hidden_dropout,
                attention_dropout=config.attention_dropout,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embedding layer."""
        self.embed_tokens = value
    
    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        past_key_values_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal attention mask.
        
        Returns a 4D mask with shape [batch, 1, seq_len, total_seq_len]
        where total_seq_len = past_key_values_length + seq_len
        """
        batch_size, seq_length = input_shape
        total_length = past_key_values_length + seq_length
        
        # Create causal mask
        causal_mask = torch.full(
            (seq_length, total_length),
            fill_value=torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=past_key_values_length + 1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, total]
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Expand attention mask: [batch, seq] -> [batch, 1, 1, seq]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_mask = expanded_mask.expand(batch_size, 1, seq_length, total_length)
            # Convert to additive mask
            inverted_mask = 1.0 - expanded_mask
            inverted_mask = inverted_mask.masked_fill(
                inverted_mask.bool(),
                torch.finfo(dtype).min,
            )
            causal_mask = causal_mask + inverted_mask
        
        return causal_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, "CausalLMOutput"]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Mask for padding [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: KV cache for each layer
            use_cache: Return updated KV cache
            output_hidden_states: Return all hidden states
            return_dict: Return CausalLMOutput instead of tuple
        
        Returns:
            CausalLMOutput or tuple containing:
            - logits: [batch, seq_len, vocab_size]
            - past_key_values: Updated KV cache
            - hidden_states: All layer outputs (if requested)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Handle past key values
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                device=device,
            ).unsqueeze(0)
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        attention_mask_4d = self._prepare_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            past_key_values_length,
            hidden_states.dtype,
            device,
        )
        
        # Get rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        next_cache = () if use_cache else None
        
        # Forward through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, present_kv = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    position_embeddings,
                    past_kv,
                    use_cache,
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )
            
            if use_cache:
                next_cache += (present_kv,)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return CausalLMOutput(
                logits=logits,
                past_key_values=next_cache if use_cache else None,
                hidden_states=all_hidden_states,
            )
        
        return (logits, next_cache, all_hidden_states)
    
    def _gradient_checkpointing_func(self, func, *args):
        """Wrapper for gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs [batch, seq]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling probability
            repetition_penalty: Penalty for repeated tokens
            eos_token_id: End of sequence token
            pad_token_id: Padding token
        
        Returns:
            Generated token IDs [batch, seq + max_new_tokens]
        """
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id
        
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for _ in range(max_new_tokens):
            # Get the input for this step
            if past_key_values is None:
                model_input = generated
            else:
                model_input = generated[:, -1:]
            
            # Forward pass
            outputs = self.forward(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]  # [batch, vocab]
            past_key_values = outputs.past_key_values
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in generated[i].unique():
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            if temperature > 0:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Replace finished sequences' tokens with pad
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, pad_token_id),
                next_token,
            )
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        
        return generated
    
    def save_pretrained(self, save_path: str):
        """Save model and config to directory."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(save_path, "config.yaml"))
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu") -> "CSTutorLLM":
        """Load model from directory."""
        import os
        
        config = ModelConfig.from_pretrained(model_path)
        model = cls(config)
        
        weights_path = os.path.join(model_path, "model.pt")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        
        return model


class CausalLMOutput:
    """Output container for causal language model."""
    
    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        hidden_states: Optional[Tuple] = None,
        loss: Optional[torch.Tensor] = None,
    ):
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.loss = loss


class CSTutorForCausalLM(CSTutorLLM):
    """
    CS Tutor LLM with loss computation for training.
    
    Extends CSTutorLLM to compute cross-entropy loss
    during training.
    """
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass with optional loss computation.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Padding mask
            labels: Target token IDs for loss computation
            ... (same as parent)
        
        Returns:
            CausalLMOutput with loss if labels provided
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        if return_dict:
            return CausalLMOutput(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
            )
        
        return (loss, outputs.logits, outputs.past_key_values, outputs.hidden_states)

