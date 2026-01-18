# GPT-Tiny

Minimal GPT-2 (124M) implementation in PyTorch, following Andrej Karpathy's "Let's build GPT-2".

## Features

- ✅ Full GPT-2 (124M) architecture implementation
- ✅ Compatible with HuggingFace pretrained weights
- ✅ Flash Attention via `F.scaled_dot_product_attention`
- ✅ Pre-Norm transformer architecture
- ✅ GELU activation with tanh approximation (matches original TF)
- ✅ Weight tying between embedding and output projection
- ✅ **Distributed training (DDP)** with multi-GPU support
- ✅ **WandB integration** for experiment tracking
- ✅ **Mixed precision training** (FP16/BF16)
- ✅ **torch.compile** optimization (PyTorch 2.0+)

## Project Structure

```
GPT-Tiny/
├── src/
│   ├── __init__.py
│   ├── model.py              # GPT-2 model implementation
│   └── data.py               # Data loading utilities
├── config/
│   ├── __init__.py
│   └── train_config.py       # Training configurations
├── scripts/
│   ├── __init__.py
│   ├── train.py              # Main training script
│   └── test_pretrained.py    # Pretrained weight loading test
├── data/
│   ├── pretrain/             # Pretraining datasets
│   │   └── .gitkeep
│   └── finetune/             # Fine-tuning datasets
│       └── .gitkeep
├── checkpoints/              # Model checkpoints
│   └── .gitkeep
├── logs/                     # Training logs
│   └── .gitkeep
├── environment.yml           # Conda environment
├── requirements.txt          # Pip requirements
├── LICENSE
└── README.md
```

## Setup

### Option 1: Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate gpt-tiny
```

### Option 2: Pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Load Pretrained Model

```python
from src.model import GPT

# Load GPT-2 (124M) pretrained weights
model = GPT.from_pretrained('gpt2')
model.eval()

# Generate text
import torch
import tiktoken

enc = tiktoken.get_encoding('gpt2')
prompt = "Hello, I'm a language model"
input_ids = torch.tensor([enc.encode(prompt)])

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8)

print(enc.decode(output_ids[0].tolist()))
```

### Test Pretrained Weights

```bash
python scripts/test_pretrained.py
```

## Training

### Data Preparation

Place your data in the appropriate directory:

```
data/
├── pretrain/           # Large-scale pretraining data
│   ├── train.bin       # Pre-tokenized binary (recommended for large data)
│   └── val.bin
└── finetune/           # Fine-tuning data
    ├── train.jsonl     # JSONL format with 'text' field
    └── val.jsonl
```

**Create binary data files:**

```python
import numpy as np
import tiktoken

enc = tiktoken.get_encoding('gpt2')

# Read and tokenize text
with open('corpus.txt', 'r') as f:
    text = f.read()

tokens = enc.encode(text)

# Save as binary
np.array(tokens, dtype=np.uint16).tofile('data/pretrain/train.bin')
```

**Or use the built-in OpenWebText preparation:**

```python
from src.data import prepare_openwebtext
prepare_openwebtext()  # Downloads and tokenizes OpenWebText
```

### Single GPU Training

```bash
# Pretrain from scratch
python scripts/train.py --config pretrain --train_data data/pretrain/train.bin

# Fine-tune from pretrained
python scripts/train.py --config finetune --train_data data/finetune/train.jsonl
```

### Multi-GPU Training (DDP)

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 scripts/train.py --config pretrain

# Multi-node (example for 2 nodes with 4 GPUs each)
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<master_ip> --master_port=29500 \
    scripts/train.py --config pretrain

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<master_ip> --master_port=29500 \
    scripts/train.py --config pretrain
```

### SLURM Cluster

```bash
#!/bin/bash
#SBATCH --job-name=gpt-tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00

srun python scripts/train.py --config pretrain
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Config preset: `pretrain`, `finetune`, `custom` | `pretrain` |
| `--train_data` | Path to training data | `data/pretrain/train.bin` |
| `--val_data` | Path to validation data | `data/pretrain/val.bin` |
| `--model_name` | Model: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` | `gpt2` |
| `--init_from` | Init: `scratch`, `pretrained`, `resume` | `scratch` |
| `--batch_size` | Micro batch size per GPU | `8` |
| `--max_steps` | Total training steps | `100000` |
| `--learning_rate` | Peak learning rate | `6e-4` |
| `--wandb_project` | WandB project name | `gpt-tiny` |
| `--no_wandb` | Disable WandB logging | `False` |
| `--no_compile` | Disable torch.compile | `False` |
| `--resume` | Resume from checkpoint path | `None` |

### Resume Training

```bash
python scripts/train.py --config pretrain --resume checkpoints/step_10000
```

## WandB Integration

Training automatically logs to [Weights & Biases](https://wandb.ai):

```bash
# Login to WandB (first time only)
wandb login

# Train with WandB
python scripts/train.py --config pretrain --wandb_project my-gpt-experiments

# Disable WandB
python scripts/train.py --config pretrain --no_wandb
```

Logged metrics:
- `train/loss`: Training loss
- `train/lr`: Learning rate
- `train/grad_norm`: Gradient norm
- `train/tokens_per_sec`: Training throughput
- `val/loss`: Validation loss

## Configuration

### Pretrain Config (Default)

```python
PretrainConfig(
    model_name='gpt2',
    init_from='scratch',
    batch_size=12,
    gradient_accumulation_steps=40,  # Effective batch: 480
    max_steps=600000,
    learning_rate=6e-4,
    warmup_steps=2000,
    dropout=0.0,
)
```

### Finetune Config

```python
FinetuneConfig(
    model_name='gpt2',
    init_from='pretrained',
    batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch: 16
    max_steps=5000,
    learning_rate=1e-5,
    warmup_steps=100,
    dropout=0.1,
)
```

### Custom Config

```python
from config import get_config

config = get_config('custom',
    train_data='my_data/train.bin',
    batch_size=16,
    max_steps=50000,
    learning_rate=3e-4,
)
```

## Architecture Details

### Model Variants

| Model | Layers | Heads | Embed | Parameters |
|-------|--------|-------|-------|------------|
| `gpt2` | 12 | 12 | 768 | 124M |
| `gpt2-medium` | 24 | 16 | 1024 | 350M |
| `gpt2-large` | 36 | 20 | 1280 | 774M |
| `gpt2-xl` | 48 | 25 | 1600 | 1558M |

### Weight Loading: The "Transpose Trap"

⚠️ **Critical**: Original GPT-2 was implemented in TensorFlow using `Conv1D` with weight shape `(input, output)`. PyTorch's `nn.Linear` uses shape `(output, input)`.

The following weights require transposition when loading:
- `attn.c_attn.weight`
- `attn.c_proj.weight`
- `mlp.c_fc.weight`
- `mlp.c_proj.weight`

This is handled automatically in `GPT.from_pretrained()`.

## Performance Tips

1. **Use BF16** on Ampere+ GPUs (A100, H100):
   ```bash
   # Default, no changes needed
   ```

2. **Enable torch.compile** (PyTorch 2.0+):
   ```bash
   # Default, use --no_compile to disable
   ```

3. **Adjust batch size and gradient accumulation**:
   ```bash
   # Larger effective batch = better training stability
   python scripts/train.py --batch_size 12 --gradient_accumulation_steps 40
   ```

4. **Use multiple workers for data loading**:
   ```python
   # In config
   num_workers=4  # Adjust based on CPU cores
   ```

## References

- [Let's build GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) by Andrej Karpathy
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [HuggingFace GPT-2](https://huggingface.co/gpt2)

## License

MIT License
