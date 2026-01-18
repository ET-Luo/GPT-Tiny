#!/usr/bin/env python3
"""
Data preparation scripts for GPT-2 training.

Provides utilities to download and preprocess popular datasets
for both pretraining and fine-tuning.

Usage:
    # Prepare OpenWebText for pretraining
    python scripts/prepare_data.py --dataset openwebtext --output data/pretrain
    
    # Prepare Alpaca for fine-tuning
    python scripts/prepare_data.py --dataset alpaca --output data/finetune
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken
from tqdm import tqdm


# =============================================================================
# Pretraining Datasets
# =============================================================================

def prepare_openwebtext(output_dir: str, val_split: float = 0.0005):
    """Download and prepare OpenWebText dataset.
    
    This is the recommended dataset for GPT-2 pretraining.
    ~40GB of high-quality web text, similar to GPT-2's original training data.
    
    Args:
        output_dir: Output directory for binary files
        val_split: Fraction of data to use for validation
    """
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing OpenWebText Dataset")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading OpenWebText from HuggingFace...")
    dataset = load_dataset('openwebtext')
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    def tokenize(example):
        tokens = enc.encode_ordinary(example['text'])
        tokens.append(enc.eot_token)  # Add end-of-text token
        return {'tokens': tokens, 'len': len(tokens)}
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized = dataset.map(
        tokenize,
        remove_columns=['text'],
        num_proc=os.cpu_count(),
        desc="Tokenizing",
    )
    
    # Split into train/val
    print(f"\nSplitting dataset (val_split={val_split})...")
    split_dataset = tokenized['train'].train_test_split(
        test_size=val_split, 
        seed=42,
        shuffle=True
    )
    
    # Save binary files
    for split, name in [('train', 'train'), ('test', 'val')]:
        print(f"\nWriting {name}.bin...")
        
        # Calculate total tokens
        total_tokens = sum(split_dataset[split]['len'])
        print(f"  Total tokens: {total_tokens:,}")
        
        # Create memory-mapped file
        arr = np.memmap(
            output_dir / f'{name}.bin',
            dtype=np.uint16,
            mode='w+',
            shape=(total_tokens,)
        )
        
        # Write tokens
        idx = 0
        for example in tqdm(split_dataset[split], desc=f"Writing {name}"):
            tokens = example['tokens']
            arr[idx:idx + len(tokens)] = tokens
            idx += len(tokens)
        
        arr.flush()
        print(f"  Saved to {output_dir / f'{name}.bin'}")
    
    print("\n✓ OpenWebText preparation complete!")


def prepare_wikipedia(output_dir: str, language: str = 'en', val_split: float = 0.001):
    """Download and prepare Wikipedia dataset.
    
    Args:
        output_dir: Output directory for binary files
        language: Wikipedia language code ('en', 'zh', etc.)
        val_split: Fraction of data to use for validation
    """
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Preparing Wikipedia ({language}) Dataset")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading Wikipedia ({language}) from HuggingFace...")
    dataset = load_dataset('wikipedia', f'20220301.{language}')
    
    enc = tiktoken.get_encoding('gpt2')
    
    def tokenize(example):
        tokens = enc.encode_ordinary(example['text'])
        tokens.append(enc.eot_token)
        return {'tokens': tokens, 'len': len(tokens)}
    
    print("\nTokenizing dataset...")
    tokenized = dataset.map(
        tokenize,
        remove_columns=['id', 'url', 'title', 'text'],
        num_proc=os.cpu_count(),
        desc="Tokenizing",
    )
    
    # Split and save
    split_dataset = tokenized['train'].train_test_split(test_size=val_split, seed=42)
    
    for split, name in [('train', 'train'), ('test', 'val')]:
        print(f"\nWriting {name}.bin...")
        total_tokens = sum(split_dataset[split]['len'])
        
        arr = np.memmap(
            output_dir / f'wikipedia_{name}.bin',
            dtype=np.uint16,
            mode='w+',
            shape=(total_tokens,)
        )
        
        idx = 0
        for example in tqdm(split_dataset[split], desc=f"Writing {name}"):
            arr[idx:idx + len(example['tokens'])] = example['tokens']
            idx += len(example['tokens'])
        
        arr.flush()
    
    print("\n✓ Wikipedia preparation complete!")


# =============================================================================
# Fine-tuning Datasets
# =============================================================================

def prepare_alpaca(output_dir: str):
    """Download and prepare Stanford Alpaca dataset.
    
    52K instruction-following examples.
    Format: instruction + input -> output
    """
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing Alpaca Dataset")
    print("=" * 70)
    
    print("\nLoading Alpaca dataset...")
    dataset = load_dataset('tatsu-lab/alpaca')
    
    def format_alpaca(example):
        """Format Alpaca example into training text."""
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']
        
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return {'text': text}
    
    print("\nFormatting examples...")
    formatted = dataset['train'].map(format_alpaca, remove_columns=['instruction', 'input', 'output'])
    
    # Split train/val
    split = formatted.train_test_split(test_size=0.05, seed=42)
    
    # Save as JSONL
    for split_name, name in [('train', 'train'), ('test', 'val')]:
        output_path = output_dir / f'alpaca_{name}.jsonl'
        print(f"\nWriting {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in split[split_name]:
                f.write(json.dumps({'text': example['text']}, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(split[split_name])} examples")
    
    print("\n✓ Alpaca preparation complete!")


def prepare_dolly(output_dir: str):
    """Download and prepare Databricks Dolly dataset.
    
    15K human-generated instruction-following examples.
    """
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing Dolly Dataset")
    print("=" * 70)
    
    print("\nLoading Dolly dataset...")
    dataset = load_dataset('databricks/databricks-dolly-15k')
    
    def format_dolly(example):
        instruction = example['instruction']
        context = example.get('context', '')
        response = example['response']
        
        if context:
            text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        return {'text': text}
    
    print("\nFormatting examples...")
    formatted = dataset['train'].map(
        format_dolly, 
        remove_columns=['instruction', 'context', 'response', 'category']
    )
    
    split = formatted.train_test_split(test_size=0.05, seed=42)
    
    for split_name, name in [('train', 'train'), ('test', 'val')]:
        output_path = output_dir / f'dolly_{name}.jsonl'
        print(f"\nWriting {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in split[split_name]:
                f.write(json.dumps({'text': example['text']}, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(split[split_name])} examples")
    
    print("\n✓ Dolly preparation complete!")


def prepare_sharegpt(output_dir: str, max_examples: Optional[int] = None):
    """Download and prepare ShareGPT-style dataset.
    
    Uses openchat/openchat_sharegpt4_dataset as a reliable alternative.
    """
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing ShareGPT Dataset")
    print("=" * 70)
    
    # Try multiple sources in order of preference
    sources = [
        ('openchat/openchat_sharegpt4_dataset', 'train'),
        ('RyokoAI/ShareGPT52K', 'train'),
    ]
    
    dataset = None
    for source, split_name in sources:
        try:
            print(f"\nTrying to load from '{source}'...")
            dataset = load_dataset(source, split=split_name)
            print(f"  ✓ Successfully loaded from {source}")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if dataset is None:
        print("\n⚠️  Could not load ShareGPT from any source.")
        print("Alternative: Download manually from https://huggingface.co/datasets/RyokoAI/ShareGPT52K")
        return
    
    def format_conversation(example):
        """Format multi-turn conversation."""
        # Handle different dataset formats
        conversations = example.get('conversations', example.get('items', []))
        
        if isinstance(conversations, str):
            # Already formatted text
            return {'text': conversations}
        
        text_parts = []
        for turn in conversations:
            if isinstance(turn, dict):
                role = turn.get('from', turn.get('role', 'user'))
                content = turn.get('value', turn.get('content', ''))
            else:
                continue
            
            if role in ['human', 'user']:
                text_parts.append(f"### Human:\n{content}")
            elif role in ['gpt', 'assistant']:
                text_parts.append(f"### Assistant:\n{content}")
        
        return {'text': '\n\n'.join(text_parts)}
    
    print("\nFormatting conversations...")
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    formatted = dataset.map(format_conversation, remove_columns=dataset.column_names)
    
    # Filter out empty examples
    formatted = formatted.filter(lambda x: len(x.get('text', '')) > 100)
    
    split = formatted.train_test_split(test_size=0.02, seed=42)
    
    for split_name, name in [('train', 'train'), ('test', 'val')]:
        output_path = output_dir / f'sharegpt_{name}.jsonl'
        print(f"\nWriting {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in split[split_name]:
                f.write(json.dumps({'text': example['text']}, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(split[split_name])} examples")
    
    print("\n✓ ShareGPT preparation complete!")


def prepare_belle_chinese(output_dir: str, max_examples: int = 100000):
    """Download and prepare BELLE Chinese instruction dataset.
    
    High-quality Chinese instruction-following data.
    """
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Preparing BELLE Chinese Dataset")
    print("=" * 70)
    
    # Try multiple BELLE dataset sources
    sources = [
        'BelleGroup/train_0.5M_CN',
        'BelleGroup/train_1M_CN', 
        'BelleGroup/generated_train_0.5M_CN',
    ]
    
    dataset = None
    for source in sources:
        try:
            print(f"\nTrying to load from '{source}'...")
            dataset = load_dataset(source, split='train')
            print(f"  ✓ Successfully loaded from {source}")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if dataset is None:
        print("\n⚠️  Could not load BELLE from any source.")
        print("Alternative: Visit https://huggingface.co/BelleGroup")
        return
    
    def format_belle(example):
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        text = f"### 指令:\n{instruction}\n\n### 回复:\n{output}"
        return {'text': text}
    
    print("\nFormatting examples...")
    # dataset is already the train split
    data = dataset.select(range(min(max_examples, len(dataset))))
    formatted = data.map(format_belle, remove_columns=data.column_names)
    
    split = formatted.train_test_split(test_size=0.02, seed=42)
    
    for split_name, name in [('train', 'train'), ('test', 'val')]:
        output_path = output_dir / f'belle_{name}.jsonl'
        print(f"\nWriting {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in split[split_name]:
                f.write(json.dumps({'text': example['text']}, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(split[split_name])} examples")
    
    print("\n✓ BELLE preparation complete!")


# =============================================================================
# Main Entry Point
# =============================================================================

DATASET_REGISTRY = {
    # Pretraining
    'openwebtext': (prepare_openwebtext, 'pretrain', 'OpenWebText - Recommended for GPT-2 pretraining'),
    'wikipedia': (prepare_wikipedia, 'pretrain', 'Wikipedia - High-quality encyclopedia text'),
    
    # Fine-tuning
    'alpaca': (prepare_alpaca, 'finetune', 'Stanford Alpaca - 52K instruction examples'),
    'dolly': (prepare_dolly, 'finetune', 'Databricks Dolly - 15K human-generated instructions'),
    'sharegpt': (prepare_sharegpt, 'finetune', 'ShareGPT - 90K ChatGPT conversations'),
    'belle': (prepare_belle_chinese, 'finetune', 'BELLE - Chinese instruction data'),
}


def list_datasets():
    """Print available datasets."""
    print("\n" + "=" * 70)
    print("Available Datasets")
    print("=" * 70)
    
    print("\n📚 Pretraining Datasets:")
    for name, (_, dtype, desc) in DATASET_REGISTRY.items():
        if dtype == 'pretrain':
            print(f"  • {name:15s} - {desc}")
    
    print("\n🎯 Fine-tuning Datasets:")
    for name, (_, dtype, desc) in DATASET_REGISTRY.items():
        if dtype == 'finetune':
            print(f"  • {name:15s} - {desc}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for GPT-2 training')
    
    parser.add_argument('--dataset', type=str, 
                        choices=list(DATASET_REGISTRY.keys()) + ['list'],
                        help='Dataset to prepare (use "list" to see all options)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data/pretrain or data/finetune)')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum examples to process (for fine-tuning datasets)')
    
    args = parser.parse_args()
    
    if args.dataset == 'list' or args.dataset is None:
        list_datasets()
        return
    
    # Get dataset info
    prepare_fn, default_dir, desc = DATASET_REGISTRY[args.dataset]
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = f'data/{default_dir}'
    
    print(f"\nPreparing: {desc}")
    print(f"Output: {output_dir}\n")
    
    # Run preparation
    if args.max_examples and 'max_examples' in prepare_fn.__code__.co_varnames:
        prepare_fn(output_dir, max_examples=args.max_examples)
    else:
        prepare_fn(output_dir)


if __name__ == '__main__':
    main()

