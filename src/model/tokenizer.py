"""
Tokenizer for CS Tutor LLM.

Supports:
- Training a BPE tokenizer from scratch
- Loading pretrained tokenizers (e.g., from LLaMA)
- Special tokens for instruction tuning
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Union

import regex as re


class CSTutorTokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer for CS Tutor LLM.
    
    Features:
    - Efficient BPE encoding/decoding
    - Special tokens for instruction format
    - Code-aware tokenization
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    
    # Instruction format tokens
    INST_START = "[INST]"
    INST_END = "[/INST]"
    SYS_START = "<<SYS>>"
    SYS_END = "<</SYS>>"
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[tuple]] = None,
        vocab_size: int = 32000,
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Token to ID mapping
            merges: BPE merge rules
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
            self.INST_START: 4,
            self.INST_END: 5,
            self.SYS_START: 6,
            self.SYS_END: 7,
        }
        
        self.pad_token_id = self.special_tokens[self.PAD_TOKEN]
        self.bos_token_id = self.special_tokens[self.BOS_TOKEN]
        self.eos_token_id = self.special_tokens[self.EOS_TOKEN]
        self.unk_token_id = self.special_tokens[self.UNK_TOKEN]
        
        # Initialize vocab and merges
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self.special_tokens.copy()
        
        self.merges = merges or []
        
        # Build reverse vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Build merge rankings
        self.merge_rankings = {merge: i for i, merge in enumerate(self.merges)}
        
        # Regex pattern for tokenization (handles code, numbers, etc.)
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    def _get_pairs(self, word: List[str]) -> set:
        """Get all adjacent pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> List[str]:
        """Apply BPE to a single token."""
        if not self.merges:
            return list(token)
        
        word = list(token)
        
        while len(word) >= 2:
            pairs = self._get_pairs(word)
            
            # Find the pair with lowest merge rank
            min_pair = None
            min_rank = float('inf')
            for pair in pairs:
                rank = self.merge_rankings.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            
            if min_pair is None or min_pair not in self.merge_rankings:
                break
            
            # Merge the pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == min_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        return word
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
        
        Returns:
            List of token IDs
        """
        # Handle special tokens first
        for special in self.special_tokens:
            if special in text:
                text = text.replace(special, f" {special} ")
        
        # Split into tokens using regex
        tokens = re.findall(self.pattern, text)
        
        # Apply BPE and convert to IDs
        token_ids = []
        
        if add_bos:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # Check if it's a special token
            if token in self.special_tokens:
                token_ids.append(self.special_tokens[token])
            else:
                # Apply BPE
                bpe_tokens = self._bpe(token)
                for bpe_token in bpe_tokens:
                    if bpe_token in self.vocab:
                        token_ids.append(self.vocab[bpe_token])
                    else:
                        token_ids.append(self.unk_token_id)
        
        if add_eos:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return "".join(tokens)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_bos: bool = True,
        add_eos: bool = True,
        padding: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """
        Tokenize text(s).
        
        Args:
            text: Input text or list of texts
            add_bos: Add BOS token
            add_eos: Add EOS token
            padding: Pad to max length
            max_length: Maximum sequence length
            return_tensors: "pt" for PyTorch tensors
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Encode all texts
        all_ids = []
        for t in texts:
            ids = self.encode(t, add_bos=add_bos, add_eos=add_eos)
            if max_length is not None:
                ids = ids[:max_length]
            all_ids.append(ids)
        
        # Pad if requested
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            if max_length is not None:
                max_len = min(max_len, max_length)
            
            attention_mask = []
            for i, ids in enumerate(all_ids):
                mask = [1] * len(ids)
                if len(ids) < max_len:
                    padding_length = max_len - len(ids)
                    ids.extend([self.pad_token_id] * padding_length)
                    mask.extend([0] * padding_length)
                all_ids[i] = ids
                attention_mask.append(mask)
        else:
            attention_mask = [[1] * len(ids) for ids in all_ids]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(all_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
        
        return {
            "input_ids": all_ids,
            "attention_mask": attention_mask,
        }
    
    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
    ):
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: Training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency for merging
        """
        vocab_size = vocab_size or self.vocab_size
        
        # Count character frequencies
        char_freq = {}
        word_freqs = {}
        
        for text in texts:
            tokens = re.findall(self.pattern, text)
            for token in tokens:
                token = token.strip()
                if token:
                    word_freqs[token] = word_freqs.get(token, 0) + 1
                    for char in token:
                        char_freq[char] = char_freq.get(char, 0) + 1
        
        # Initialize vocab with characters
        self.vocab = self.special_tokens.copy()
        for char in sorted(char_freq.keys()):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Convert words to character lists with frequencies
        word_splits = {word: list(word) for word in word_freqs}
        
        # Perform BPE merges
        self.merges = []
        while len(self.vocab) < vocab_size:
            # Count pairs
            pair_freq = {}
            for word, freq in word_freqs.items():
                split = word_splits[word]
                for i in range(len(split) - 1):
                    pair = (split[i], split[i+1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq
            
            if not pair_freq:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            if pair_freq[best_pair] < min_frequency:
                break
            
            # Merge the pair
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            
            # Update word splits
            for word in word_splits:
                split = word_splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i+1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_splits[word] = new_split
        
        # Update reverse vocab and merge rankings
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merge_rankings = {merge: i for i, merge in enumerate(self.merges)}
    
    def save(self, save_path: str):
        """Save tokenizer to directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocab
        with open(save_path / "vocab.json", 'w') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        with open(save_path / "merges.txt", 'w') as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(save_path / "tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, tokenizer_path: str) -> "CSTutorTokenizer":
        """Load tokenizer from directory."""
        tokenizer_path = Path(tokenizer_path)
        
        # Load vocab
        with open(tokenizer_path / "vocab.json", 'r') as f:
            vocab = json.load(f)
        
        # Load merges
        merges = []
        merges_file = tokenizer_path / "merges.txt"
        if merges_file.exists():
            with open(merges_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 2:
                            merges.append((parts[0], parts[1]))
        
        # Load config
        config_file = tokenizer_path / "tokenizer_config.json"
        vocab_size = 32000
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                vocab_size = config.get("vocab_size", 32000)
        
        return cls(vocab=vocab, merges=merges, vocab_size=vocab_size)


def load_llama_tokenizer(model_path: str) -> CSTutorTokenizer:
    """
    Load a LLaMA-compatible tokenizer.
    
    This is a convenience function for loading tokenizers
    from LLaMA-style models.
    
    Args:
        model_path: Path to model directory with tokenizer files
    
    Returns:
        CSTutorTokenizer instance
    """
    # Try loading from sentencepiece model
    try:
        import sentencepiece as spm
        
        sp_model = os.path.join(model_path, "tokenizer.model")
        if os.path.exists(sp_model):
            sp = spm.SentencePieceProcessor()
            sp.Load(sp_model)
            
            # Build vocab from sentencepiece
            vocab = {}
            for i in range(sp.GetPieceSize()):
                vocab[sp.IdToPiece(i)] = i
            
            tokenizer = CSTutorTokenizer(vocab=vocab, vocab_size=sp.GetPieceSize())
            return tokenizer
    except ImportError:
        pass
    
    # Fall back to our format
    return CSTutorTokenizer.from_pretrained(model_path)

