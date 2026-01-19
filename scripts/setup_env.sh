#!/bin/bash
# =============================================================================
# GPT-Tiny Environment Setup Script
# =============================================================================
# Run this script before training to configure environment variables.
# Usage: source scripts/setup_env.sh
# =============================================================================

# HuggingFace cache directory (large storage)
export HF_HOME=/hpcwork/qw765731/cache/huggingface
export HF_DATASETS_CACHE=/hpcwork/qw765731/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/hpcwork/qw765731/cache/huggingface/hub

# PyTorch settings
export TORCH_HOME=/hpcwork/qw765731/cache/torch

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# CUDA settings (adjust based on your cluster)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# WandB settings (optional: set offline mode for clusters without internet)
# export WANDB_MODE=offline
# export WANDB_DIR=/hpcwork/qw765731/logs/wandb

echo "✓ Environment configured:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  TORCH_HOME=$TORCH_HOME"
echo ""
echo "Storage locations:"
echo "  Data:        /home/qw765731/projects/GPT/GPT-Tiny/data -> /hpcwork/qw765731/datasets"
echo "  Checkpoints: /home/qw765731/projects/GPT/GPT-Tiny/checkpoints -> /hpcwork/qw765731/models/checkpoints"
echo "  HF Cache:    ~/.cache/huggingface -> /hpcwork/qw765731/cache/huggingface"








