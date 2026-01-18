#!/usr/bin/env python3
"""
GPT-2 Training Script with Distributed Training and WandB Logging.

Features:
- Single GPU and Multi-GPU (DDP) training
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Cosine learning rate schedule with warmup
- WandB integration for experiment tracking
- Checkpoint saving and resuming
- torch.compile optimization (PyTorch 2.0+)

Usage:
    # Single GPU
    python scripts/train.py --config pretrain
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 scripts/train.py --config pretrain
    
    # SLURM cluster
    srun python scripts/train.py --config pretrain
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from src.model import GPT, GPT2Config
from src.data import create_dataloader
from config.train_config import get_config, TrainConfig


# =============================================================================
# Distributed Training Setup
# =============================================================================

def setup_distributed():
    """Initialize distributed training environment.
    
    Supports:
    - torchrun (recommended)
    - SLURM
    - Manual launch
    
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    # Check if distributed training is requested
    if 'RANK' in os.environ:
        # torchrun sets these
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # Set master address for SLURM
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get('SLURM_NODELIST', 'localhost').split(',')[0]
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    else:
        # Single GPU
        rank = 0
        local_rank = 0
        world_size = 1
    
    # Initialize process group if distributed
    if world_size > 1:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def get_lr(step: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with linear warmup.
    
    Args:
        step: Current training step
        config: Training configuration
        
    Returns:
        Learning rate for this step
    """
    # Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    
    # Cosine decay to min_lr
    if step > config.max_steps:
        return config.min_lr
    
    # Cosine annealing
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges from 1 to 0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# =============================================================================
# Training Loop
# =============================================================================

@torch.no_grad()
def evaluate(model, val_loader, config, ctx, device):
    """Run evaluation on validation set.
    
    Args:
        model: GPT model
        val_loader: Validation data loader
        config: Training configuration
        ctx: Autocast context
        device: Device to run on
        
    Returns:
        Average validation loss
    """
    model.eval()
    losses = []
    
    for i, batch in enumerate(val_loader):
        if i >= config.eval_steps:
            break
            
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        with ctx:
            _, loss = model(input_ids, labels=labels)
        
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def save_checkpoint(model, optimizer, scaler, step, config, rank, is_best=False):
    """Save training checkpoint.
    
    Args:
        model: GPT model (may be wrapped in DDP)
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        step: Current step
        config: Training configuration
        rank: Process rank
        is_best: Whether this is the best model so far
    """
    if not is_main_process(rank):
        return
    
    # Get raw model (unwrap DDP if necessary)
    raw_model = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'step': step,
        'config': config.to_dict(),
    }
    
    # Save checkpoint
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_path = checkpoint_dir / f'step_{step}'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path / 'checkpoint.pt')
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best'
        best_path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, best_path / 'checkpoint.pt')
        print(f"Saved best model to {best_path}")
    
    # Clean up old checkpoints
    checkpoints = sorted(checkpoint_dir.glob('step_*'), key=lambda x: int(x.name.split('_')[1]))
    while len(checkpoints) > config.keep_last_n:
        old_checkpoint = checkpoints.pop(0)
        import shutil
        shutil.rmtree(old_checkpoint)
        print(f"Removed old checkpoint: {old_checkpoint}")


def load_checkpoint(model, optimizer, scaler, config, device):
    """Load checkpoint if resuming training.
    
    Returns:
        Starting step number
    """
    if config.init_from != 'resume' or config.resume_checkpoint is None:
        return 0
    
    checkpoint_path = Path(config.resume_checkpoint) / 'checkpoint.pt'
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return 0
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get raw model (unwrap DDP if necessary)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(checkpoint['model'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scaler is not None and checkpoint.get('scaler') is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    
    start_step = checkpoint['step']
    print(f"Resumed from step {start_step}")
    
    return start_step


def train(config: TrainConfig):
    """Main training function.
    
    Args:
        config: Training configuration
    """
    # =========================================================================
    # Setup
    # =========================================================================
    
    # Initialize distributed training
    rank, local_rank, world_size = setup_distributed()
    is_main = is_main_process(rank)
    
    # Set device
    if torch.cuda.is_available():
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + rank)
    
    # =========================================================================
    # WandB Setup
    # =========================================================================
    
    if config.wandb_log and is_main:
        import wandb
        
        run_name = config.wandb_run_name or f"gpt2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=config.to_dict(),
            resume='allow',
        )
        print(f"WandB initialized: {wandb.run.url}")
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    
    if is_main:
        print("=" * 70)
        print("Initializing model...")
        print("=" * 70)
    
    # Model configuration
    model_configs = {
        'gpt2':        GPT2Config(n_layer=12, n_head=12, n_embd=768, dropout=config.dropout),
        'gpt2-medium': GPT2Config(n_layer=24, n_head=16, n_embd=1024, dropout=config.dropout),
        'gpt2-large':  GPT2Config(n_layer=36, n_head=20, n_embd=1280, dropout=config.dropout),
        'gpt2-xl':     GPT2Config(n_layer=48, n_head=25, n_embd=1600, dropout=config.dropout),
    }
    
    model_config = model_configs[config.model_name]
    
    # Initialize model
    if config.init_from == 'scratch':
        if is_main:
            print("Initializing model from scratch...")
        model = GPT(model_config)
    elif config.init_from == 'pretrained':
        if is_main:
            print(f"Loading pretrained weights from '{config.model_name}'...")
        model = GPT.from_pretrained(config.model_name)
        # Update dropout for fine-tuning
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = config.dropout
    elif config.init_from == 'resume':
        if is_main:
            print("Will resume from checkpoint...")
        model = GPT(model_config)
    else:
        raise ValueError(f"Unknown init_from: {config.init_from}")
    
    model = model.to(device)
    
    # Compile model (PyTorch 2.0+)
    if config.compile and hasattr(torch, 'compile'):
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Wrap in DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # =========================================================================
    # Optimizer Setup
    # =========================================================================
    
    # Separate parameters for weight decay
    # Don't apply weight decay to biases and LayerNorm weights
    decay_params = []
    no_decay_params = []
    
    raw_model = model.module if hasattr(model, 'module') else model
    
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'ln_' in name or 'wpe' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=torch.cuda.is_available(),  # Fused optimizer for GPU
    )
    
    # =========================================================================
    # Mixed Precision Setup
    # =========================================================================
    
    # Determine dtype
    if config.dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        scaler = None  # BF16 doesn't need gradient scaling
    elif config.dtype == 'float16':
        dtype = torch.float16
        scaler = GradScaler()
    else:
        dtype = torch.float32
        scaler = None
    
    # Autocast context
    ctx = torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=dtype)
    
    if is_main:
        print(f"Using dtype: {dtype}")
    
    # =========================================================================
    # Data Setup
    # =========================================================================
    
    if is_main:
        print("\n" + "=" * 70)
        print("Loading data...")
        print("=" * 70)
    
    train_loader = create_dataloader(
        config.train_data,
        batch_size=config.batch_size,
        block_size=config.block_size,
        num_workers=config.num_workers,
        shuffle=True,
        distributed=(world_size > 1),
        world_size=world_size,
        rank=rank,
    )
    
    val_loader = None
    if config.val_data and Path(config.val_data).exists():
        val_loader = create_dataloader(
            config.val_data,
            batch_size=config.batch_size,
            block_size=config.block_size,
            num_workers=config.num_workers,
            shuffle=False,
            distributed=(world_size > 1),
            world_size=world_size,
            rank=rank,
        )
    
    # =========================================================================
    # Resume from Checkpoint
    # =========================================================================
    
    start_step = load_checkpoint(model, optimizer, scaler, config, device)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    if is_main:
        print("\n" + "=" * 70)
        print("Starting training...")
        print(f"  Total steps: {config.max_steps:,}")
        print(f"  Batch size per GPU: {config.batch_size}")
        print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps * world_size}")
        print(f"  World size: {world_size}")
        print("=" * 70 + "\n")
    
    model.train()
    
    # Training state
    step = start_step
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    tokens_processed = 0
    t0 = time.time()
    
    # Accumulation tracking
    micro_step = 0
    accumulated_loss = 0.0
    
    while step < config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reset iterator at epoch boundary
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(step)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        with ctx:
            _, loss = model(input_ids, labels=labels)
            loss = loss / config.gradient_accumulation_steps
        
        accumulated_loss += loss.item()
        tokens_processed += input_ids.numel()
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        micro_step += 1
        
        # Gradient accumulation
        if micro_step < config.gradient_accumulation_steps:
            continue
        
        # Gradient clipping
        if config.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = 0.0
        
        # Update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Synchronize for timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        # Logging
        if step % config.log_interval == 0 and is_main:
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0
            
            log_dict = {
                'train/loss': accumulated_loss,
                'train/lr': lr,
                'train/grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                'train/tokens_per_sec': tokens_per_sec,
                'train/step': step,
            }
            
            print(f"Step {step:6d} | Loss: {accumulated_loss:.4f} | LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.0f} | Time: {dt*1000:.1f}ms")
            
            if config.wandb_log:
                import wandb
                wandb.log(log_dict, step=step)
        
        # Reset accumulation
        accumulated_loss = 0.0
        tokens_processed = 0
        micro_step = 0
        
        # Evaluation
        if val_loader is not None and step % config.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_loader, config, ctx, device)
            
            if is_main:
                print(f"  Validation loss: {val_loss:.4f}")
                
                if config.wandb_log:
                    import wandb
                    wandb.log({'val/loss': val_loss}, step=step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scaler, step, config, rank, is_best=True)
        
        # Checkpoint saving
        if step % config.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, scaler, step, config, rank)
        
        step += 1
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    # Final checkpoint
    if is_main:
        save_checkpoint(model, optimizer, scaler, step, config, rank)
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)
        
        if config.wandb_log:
            import wandb
            wandb.finish()
    
    cleanup_distributed()


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT-2 model')
    
    # Configuration
    parser.add_argument('--config', type=str, default='pretrain',
                        choices=['pretrain', 'finetune', 'custom'],
                        help='Configuration preset')
    
    # Override common settings
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--val_data', type=str, help='Path to validation data')
    parser.add_argument('--model_name', type=str, help='Model variant')
    parser.add_argument('--init_from', type=str, help='Initialization method')
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--max_steps', type=int, help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, help='Peak learning rate')
    parser.add_argument('--wandb_project', type=str, help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, help='WandB run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build config with overrides
    overrides = {}
    
    if args.train_data:
        overrides['train_data'] = args.train_data
    if args.val_data:
        overrides['val_data'] = args.val_data
    if args.model_name:
        overrides['model_name'] = args.model_name
    if args.init_from:
        overrides['init_from'] = args.init_from
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.max_steps:
        overrides['max_steps'] = args.max_steps
    if args.learning_rate:
        overrides['learning_rate'] = args.learning_rate
    if args.wandb_project:
        overrides['wandb_project'] = args.wandb_project
    if args.wandb_run_name:
        overrides['wandb_run_name'] = args.wandb_run_name
    if args.no_wandb:
        overrides['wandb_log'] = False
    if args.no_compile:
        overrides['compile'] = False
    if args.checkpoint_dir:
        overrides['checkpoint_dir'] = args.checkpoint_dir
    if args.resume:
        overrides['init_from'] = 'resume'
        overrides['resume_checkpoint'] = args.resume
    
    config = get_config(args.config, **overrides)
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()

