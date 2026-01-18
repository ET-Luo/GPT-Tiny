#!/bin/bash
# =============================================================================
# Download Datasets Script
# =============================================================================
# Usage: bash scripts/download_datasets.sh [openwebtext|sharegpt|belle|all]
# =============================================================================

set -e

# Setup environment
export HF_HOME=/hpcwork/qw765731/cache/huggingface
export HF_DATASETS_CACHE=/hpcwork/qw765731/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/hpcwork/qw765731/cache/huggingface/hub

# Python path
PYTHON=/rwthfs/rz/cluster/home/qw765731/miniforge3/envs/gpt-tiny/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "GPT-Tiny Dataset Download Script"
echo "=============================================="
echo "HF_HOME: $HF_HOME"
echo "Project: $PROJECT_DIR"
echo ""

download_openwebtext() {
    echo "Downloading OpenWebText..."
    $PYTHON "$SCRIPT_DIR/prepare_data.py" --dataset openwebtext --output "$PROJECT_DIR/data/pretrain"
}

download_sharegpt() {
    echo "Downloading ShareGPT..."
    $PYTHON "$SCRIPT_DIR/prepare_data.py" --dataset sharegpt --output "$PROJECT_DIR/data/finetune"
}

download_belle() {
    echo "Downloading BELLE..."
    $PYTHON "$SCRIPT_DIR/prepare_data.py" --dataset belle --output "$PROJECT_DIR/data/finetune"
}

case "${1:-all}" in
    openwebtext)
        download_openwebtext
        ;;
    sharegpt)
        download_sharegpt
        ;;
    belle)
        download_belle
        ;;
    all)
        download_openwebtext
        download_sharegpt
        download_belle
        ;;
    *)
        echo "Usage: $0 [openwebtext|sharegpt|belle|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="


