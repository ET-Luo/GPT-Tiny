# =============================================================================
# GPT-Tiny Justfile
# =============================================================================
# 常用命令集合: just <command> [args...]
# 查看所有命令: just --list
# =============================================================================

# 默认命令：显示帮助
default:
    @just --list

# =============================================================================
# 环境配置
# =============================================================================

# 加载 miniforge3 自带的激活脚本
env-activate:
    source $HOME/miniforge3/bin/activate


# 配置环境变量
setup-env:
    source scripts/setup_env.sh

# WandB 登录
wandb-login:
    wandb login

# =============================================================================
# 数据准备
# =============================================================================

# 列出可用数据集
data-list:
    python scripts/prepare_data.py --dataset list

# 下载并准备 OpenWebText (预训练)
data-openwebtext:
    python scripts/prepare_data.py --dataset openwebtext --output data/pretrain

# 下载并准备 Wikipedia (预训练)
data-wikipedia:
    python scripts/prepare_data.py --dataset wikipedia --output data/pretrain

# 下载并准备 Alpaca (微调)
data-alpaca:
    python scripts/prepare_data.py --dataset alpaca --output data/finetune

# 下载并准备 Dolly (微调)
data-dolly:
    python scripts/prepare_data.py --dataset dolly --output data/finetune

# 下载并准备 ShareGPT (微调)
data-sharegpt:
    python scripts/prepare_data.py --dataset sharegpt --output data/finetune

# 下载并准备 BELLE 中文数据 (微调)
data-belle:
    python scripts/prepare_data.py --dataset belle --output data/finetune

# 下载所有数据集
data-all:
    bash scripts/download_datasets.sh all

# =============================================================================
# 测试
# =============================================================================

# 测试预训练权重加载
test-pretrained:
    python scripts/test_pretrained.py

# =============================================================================
# 单 GPU 训练
# =============================================================================

# 从头预训练 (单 GPU)
train-pretrain:
    python scripts/train.py --config pretrain

# 微调 (单 GPU)
train-finetune:
    python scripts/train.py --config finetune

# 自定义训练数据预训练
train-pretrain-data data_path:
    python scripts/train.py --config pretrain --train_data {{data_path}}

# 自定义训练数据微调
train-finetune-data data_path:
    python scripts/train.py --config finetune --train_data {{data_path}}

# 预训练 (禁用 WandB)
train-pretrain-nowb:
    python scripts/train.py --config pretrain --no_wandb

# 微调 (禁用 WandB)
train-finetune-nowb:
    python scripts/train.py --config finetune --no_wandb

# 从检查点恢复训练
train-resume checkpoint_path:
    python scripts/train.py --config pretrain --resume {{checkpoint_path}}

# 带自定义参数的训练
train config="pretrain" *args="":
    python scripts/train.py --config {{config}} {{args}}

# =============================================================================
# 多 GPU 训练 (DDP)
# =============================================================================

# 4 GPU 预训练
train-ddp-4gpu:
    torchrun --nproc_per_node=4 scripts/train.py --config pretrain

# 多 GPU 预训练 (自定义 GPU 数量)
train-ddp gpus="4":
    torchrun --nproc_per_node={{gpus}} scripts/train.py --config pretrain

# 多 GPU 微调
train-ddp-finetune gpus="4":
    torchrun --nproc_per_node={{gpus}} scripts/train.py --config finetune

# 多 GPU 训练 (禁用 WandB)
train-ddp-nowb gpus="4":
    torchrun --nproc_per_node={{gpus}} scripts/train.py --config pretrain --no_wandb

# 多 GPU 带自定义参数
train-ddp-custom gpus="4" config="pretrain" *args="":
    torchrun --nproc_per_node={{gpus}} scripts/train.py --config {{config}} {{args}}

# =============================================================================
# 模型变体训练
# =============================================================================

# 训练 GPT-2 Medium (350M)
train-medium:
    python scripts/train.py --config pretrain --model_name gpt2-medium

# 训练 GPT-2 Large (774M)
train-large:
    python scripts/train.py --config pretrain --model_name gpt2-large

# 训练 GPT-2 XL (1558M)
train-xl:
    python scripts/train.py --config pretrain --model_name gpt2-xl

# 从预训练权重微调指定模型
finetune-model model_name:
    python scripts/train.py --config finetune --model_name {{model_name}} --init_from pretrained

# =============================================================================
# 检查点管理
# =============================================================================

# 列出所有检查点
checkpoints-list:
    @ls -la checkpoints/ 2>/dev/null || echo "checkpoints 目录为空或不存在"

# 查看最新检查点
checkpoint-latest:
    @ls -td checkpoints/step_* 2>/dev/null | head -1 || echo "没有找到检查点"

# 清理旧检查点 (保留最近 n 个)
checkpoints-clean n="3":
    @echo "保留最近 {{n}} 个检查点..."
    @ls -td checkpoints/step_* 2>/dev/null | tail -n +$(({{n}}+1)) | xargs rm -rf 2>/dev/null || echo "无需清理"

# =============================================================================
# 日志管理
# =============================================================================

# 查看最新日志
logs-latest:
    @ls -t logs/*.log 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "logs 目录为空"

# 列出所有日志
logs-list:
    @ls -la logs/ 2>/dev/null || echo "logs 目录为空或不存在"

# 清理日志
logs-clean:
    rm -rf logs/*.log 2>/dev/null || echo "无需清理"

# =============================================================================
# 实用工具
# =============================================================================

# 清理 Python 缓存
clean-pycache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 完全清理 (缓存 + 日志 + 旧检查点)
clean-all: clean-pycache logs-clean
    @echo "清理完成!"

