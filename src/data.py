"""
Data loading utilities for GPT-2 training.

Supports:
1. Pre-tokenized binary files (memory-mapped for large datasets)
2. Raw text files with on-the-fly tokenization
3. HuggingFace datasets integration
4. Distributed data loading with proper sharding
"""

import os
import json
from pathlib import Path
from typing import Optional, Iterator, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken


# =============================================================================
# PreTokenizedDataset: Memory-mapped binary dataset
# =============================================================================

class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized binary files.
    
    Expects data stored as numpy memmap files with dtype uint16.
    This is the most efficient format for large-scale pretraining.
    
    Binary file format:
    - Flat array of token IDs
    - dtype: uint16 (supports vocab size up to 65535)
    
    To create binary files:
    ```python
    import numpy as np
    import tiktoken
    
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    np.array(tokens, dtype=np.uint16).tofile('data.bin')
    ```
    """
    
    def __init__(
        self,
        data_path: str,
        block_size: int = 1024,
        dtype: np.dtype = np.uint16,
    ):
        """
        Args:
            data_path: Path to binary file (.bin)
            block_size: Sequence length for training
            dtype: Data type of tokens (default: uint16)
        """
        self.block_size = block_size
        
        # Memory-map the file for efficient random access
        self.data = np.memmap(data_path, dtype=dtype, mode='r')
        self.n_tokens = len(self.data)
        
        # Number of complete sequences we can form
        self.n_samples = (self.n_tokens - 1) // block_size
        
        print(f"Loaded {self.n_tokens:,} tokens from {data_path}")
        print(f"  -> {self.n_samples:,} training samples of length {block_size}")
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        # Get a sequence of block_size + 1 tokens (input + target)
        start = idx * self.block_size
        end = start + self.block_size + 1
        
        # Handle edge case at the end of data
        if end > self.n_tokens:
            end = self.n_tokens
            start = end - self.block_size - 1
        
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        
        return {
            'input_ids': chunk[:-1],
            'labels': chunk[1:],
        }


# =============================================================================
# TextDataset: Raw text dataset with on-the-fly tokenization
# =============================================================================

class TextDataset(Dataset):
    """Dataset for raw text files with on-the-fly tokenization.
    
    Suitable for smaller datasets or fine-tuning.
    For large-scale pretraining, use PreTokenizedDataset instead.
    """
    
    def __init__(
        self,
        data_path: str,
        block_size: int = 1024,
        encoding: str = 'gpt2',
    ):
        """
        Args:
            data_path: Path to text file or directory of text files
            block_size: Sequence length for training
            encoding: Tiktoken encoding name
        """
        self.block_size = block_size
        self.enc = tiktoken.get_encoding(encoding)
        
        # Load and tokenize all text
        data_path = Path(data_path)
        
        if data_path.is_file():
            text = data_path.read_text(encoding='utf-8')
        else:
            # Concatenate all text files in directory
            texts = []
            for txt_file in sorted(data_path.glob('**/*.txt')):
                texts.append(txt_file.read_text(encoding='utf-8'))
            text = '\n\n'.join(texts)
        
        # Tokenize
        self.tokens = self.enc.encode(text)
        self.n_tokens = len(self.tokens)
        self.n_samples = (self.n_tokens - 1) // block_size
        
        print(f"Tokenized {len(text):,} characters -> {self.n_tokens:,} tokens")
        print(f"  -> {self.n_samples:,} training samples of length {block_size}")
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        start = idx * self.block_size
        end = start + self.block_size + 1
        
        if end > self.n_tokens:
            end = self.n_tokens
            start = end - self.block_size - 1
            
        chunk = torch.tensor(self.tokens[start:end], dtype=torch.long)
        
        return {
            'input_ids': chunk[:-1],
            'labels': chunk[1:],
        }


# =============================================================================
# JSONLDataset: JSON Lines dataset for instruction tuning
# =============================================================================

class JSONLDataset(Dataset):
    """Dataset for JSONL files (instruction tuning, chat, etc).
    
    Expected format:
    {"text": "..."}  or
    {"prompt": "...", "completion": "..."} or
    {"messages": [{"role": "user", "content": "..."}, ...]}
    """
    
    def __init__(
        self,
        data_path: str,
        block_size: int = 1024,
        encoding: str = 'gpt2',
        text_field: str = 'text',
    ):
        """
        Args:
            data_path: Path to JSONL file
            block_size: Maximum sequence length
            encoding: Tiktoken encoding name
            text_field: Field name containing text (or 'auto' to detect)
        """
        self.block_size = block_size
        self.enc = tiktoken.get_encoding(encoding)
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = self._extract_text(item, text_field)
                    if text:
                        tokens = self.enc.encode(text)
                        # Truncate to block_size
                        if len(tokens) > block_size:
                            tokens = tokens[:block_size]
                        self.samples.append(tokens)
        
        print(f"Loaded {len(self.samples):,} samples from {data_path}")
        
    def _extract_text(self, item: dict, text_field: str) -> Optional[str]:
        """Extract text from various JSONL formats."""
        if text_field in item:
            return item[text_field]
        elif 'prompt' in item and 'completion' in item:
            return item['prompt'] + item['completion']
        elif 'messages' in item:
            # Chat format
            texts = []
            for msg in item['messages']:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                texts.append(f"{role}: {content}")
            return '\n'.join(texts)
        elif 'input' in item and 'output' in item:
            return item['input'] + item['output']
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        tokens = self.samples[idx]
        
        # Pad if necessary
        if len(tokens) < self.block_size:
            # Pad with -100 (ignored in loss)
            padding = [-100] * (self.block_size - len(tokens))
            labels = tokens[1:] + [-100]  # Shift for next-token prediction
            input_ids = tokens + [0] * (self.block_size - len(tokens))
            labels = labels + padding[:-1]
        else:
            input_ids = tokens[:-1]
            labels = tokens[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloader(
    data_path: str,
    batch_size: int = 8,
    block_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create a DataLoader with appropriate dataset based on file type.
    
    Args:
        data_path: Path to data file or directory
        batch_size: Batch size per GPU
        block_size: Sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data (ignored if distributed)
        distributed: Whether using distributed training
        world_size: Total number of processes (for distributed)
        rank: Current process rank (for distributed)
        
    Returns:
        DataLoader instance
    """
    data_path = Path(data_path)
    
    # Determine dataset type based on file extension
    if data_path.suffix == '.bin':
        dataset = PreTokenizedDataset(str(data_path), block_size=block_size)
    elif data_path.suffix == '.jsonl':
        dataset = JSONLDataset(str(data_path), block_size=block_size)
    elif data_path.suffix == '.txt' or data_path.is_dir():
        dataset = TextDataset(str(data_path), block_size=block_size)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for consistent batch size
    )


# =============================================================================
# Data Preparation Utilities
# =============================================================================

def prepare_openwebtext(output_dir: str = 'data/pretrain/openwebtext'):
    """Download and prepare OpenWebText dataset.
    
    Creates train.bin and val.bin files.
    """
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading OpenWebText dataset...")
    dataset = load_dataset('openwebtext', trust_remote_code=True)
    
    enc = tiktoken.get_encoding('gpt2')
    
    def tokenize(example):
        return {'tokens': enc.encode(example['text'])}
    
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize,
        remove_columns=['text'],
        num_proc=os.cpu_count(),
        desc="Tokenizing",
    )
    
    # Split into train/val
    split_dataset = tokenized['train'].train_test_split(test_size=0.0005, seed=42)
    
    for split, name in [('train', 'train'), ('test', 'val')]:
        print(f"Writing {name}.bin...")
        tokens = []
        for example in split_dataset[split]:
            tokens.extend(example['tokens'])
        
        tokens = np.array(tokens, dtype=np.uint16)
        tokens.tofile(output_dir / f'{name}.bin')
        print(f"  -> {len(tokens):,} tokens saved to {name}.bin")
    
    print("Done!")


if __name__ == "__main__":
    # Quick test
    import tempfile
    
    # Create a small test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Hello, I'm a language model. " * 100)
        test_path = f.name
    
    print("Testing TextDataset...")
    dataset = TextDataset(test_path, block_size=64)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    
    # Test dataloader
    loader = create_dataloader(test_path, batch_size=4, block_size=64, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    
    os.unlink(test_path)
    print("\n✓ All data tests passed!")

