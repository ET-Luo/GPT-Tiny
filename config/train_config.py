"""
Training configuration for GPT-2.

Provides dataclass-based configuration with sensible defaults
for both pretraining and fine-tuning scenarios.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class TrainConfig:
    """Training configuration.
    
    Organized into logical groups:
    - Data settings
    - Model settings  
    - Optimization settings
    - Distributed training settings
    - Logging settings
    - Checkpoint settings
    """
    
    # =========================================================================
    # Data Settings
    # =========================================================================
    train_data: str = "data/pretrain/train.bin"
    val_data: Optional[str] = "data/pretrain/val.bin"
    block_size: int = 1024  # Sequence length
    
    # =========================================================================
    # Model Settings
    # =========================================================================
    model_name: str = "gpt2"  # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
    init_from: str = "scratch"  # 'scratch', 'pretrained', 'resume'
    resume_checkpoint: Optional[str] = None  # Path to checkpoint for resume
    dropout: float = 0.0  # Dropout probability (0.0 for pretraining)
    
    # =========================================================================
    # Optimization Settings
    # =========================================================================
    batch_size: int = 8  # Micro batch size per GPU
    gradient_accumulation_steps: int = 8  # Effective batch = batch_size * grad_accum * world_size
    max_steps: int = 100000  # Total training steps
    warmup_steps: int = 2000  # Linear warmup steps
    
    # Learning rate schedule (cosine decay with warmup)
    learning_rate: float = 6e-4  # Peak learning rate
    min_lr: float = 6e-5  # Minimum learning rate (10% of peak)
    weight_decay: float = 0.1  # AdamW weight decay
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.95  # Adam beta2 (GPT-3 uses 0.95)
    grad_clip: float = 1.0  # Gradient clipping (0 = disabled)
    
    # =========================================================================
    # Distributed Training Settings
    # =========================================================================
    distributed: bool = False  # Enable distributed training (auto-detected)
    backend: str = "nccl"  # DDP backend ('nccl' for GPU, 'gloo' for CPU)
    
    # =========================================================================
    # Mixed Precision Settings
    # =========================================================================
    dtype: str = "bfloat16"  # 'float32', 'float16', 'bfloat16'
    compile: bool = True  # Use torch.compile (PyTorch 2.0+)
    
    # =========================================================================
    # Logging Settings
    # =========================================================================
    wandb_project: str = "gpt-tiny"
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_log: bool = True  # Enable WandB logging
    log_interval: int = 10  # Log every N steps
    eval_interval: int = 500  # Evaluate every N steps
    eval_steps: int = 100  # Number of eval steps
    
    # =========================================================================
    # Checkpoint Settings
    # =========================================================================
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000  # Save checkpoint every N steps
    keep_last_n: int = 3  # Keep only last N checkpoints
    
    # =========================================================================
    # System Settings
    # =========================================================================
    seed: int = 42
    num_workers: int = 4  # DataLoader workers
    device: str = "cuda"  # 'cuda', 'cpu', 'mps'
    
    def __post_init__(self):
        """Validate and process configuration."""
        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate model name
        valid_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid model_name: {self.model_name}. Must be one of {valid_models}")
        
        # Validate dtype
        valid_dtypes = ['float32', 'float16', 'bfloat16']
        if self.dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be one of {valid_dtypes}")
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size across all GPUs and accumulation steps."""
        world_size = 1  # Will be updated in distributed setting
        return self.batch_size * self.gradient_accumulation_steps * world_size
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class PretrainConfig(TrainConfig):
    """Configuration preset for pretraining from scratch."""
    
    init_from: str = "scratch"
    dropout: float = 0.0
    max_steps: int = 600000
    batch_size: int = 12
    gradient_accumulation_steps: int = 40  # ~480 effective batch size
    learning_rate: float = 6e-4
    warmup_steps: int = 2000


@dataclass 
class FinetuneConfig(TrainConfig):
    """Configuration preset for fine-tuning pretrained model."""
    
    init_from: str = "pretrained"
    dropout: float = 0.1
    max_steps: int = 5000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    train_data: str = "data/finetune/train.jsonl"
    val_data: Optional[str] = "data/finetune/val.jsonl"


# =============================================================================
# Configuration Factory
# =============================================================================

def get_config(config_type: str = "pretrain", **overrides) -> TrainConfig:
    """Get a configuration with optional overrides.
    
    Args:
        config_type: 'pretrain', 'finetune', or 'custom'
        **overrides: Override any config parameter
        
    Returns:
        TrainConfig instance
    """
    if config_type == "pretrain":
        config = PretrainConfig()
    elif config_type == "finetune":
        config = FinetuneConfig()
    else:
        config = TrainConfig()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    
    return config


if __name__ == "__main__":
    # Test configurations
    print("=" * 60)
    print("Pretrain Config:")
    print("=" * 60)
    pretrain_config = get_config("pretrain")
    for k, v in pretrain_config.to_dict().items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Finetune Config:")
    print("=" * 60)
    finetune_config = get_config("finetune")
    for k, v in finetune_config.to_dict().items():
        print(f"  {k}: {v}")

