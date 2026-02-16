#!/bin/bash
#SBATCH --job-name=kg_rl_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH -o ./training_%j.out

# ===================================================================
# SLURM Template for Knowledge Graph-Guided RL Training
# 
# Replace the following placeholders:
# - <YOUR_PARTITION>: Your cluster partition name
# - <YOUR_EMAIL>: Your email for job notifications
# - <YOUR_CONDA_ENV>: Your conda environment name
# - <PATH_TO_REPO>: Path to this repository
# - <PATH_TO_DATASET>: Path to your training dataset
# - <PATH_TO_OUTPUT>: Path to save model checkpoints
# ===================================================================

# Load required modules (adjust for your cluster)
module purge
module load anaconda3/2024.6
module load cudatoolkit/12.8
module load gcc/11

# Activate your conda environment
conda activate <YOUR_CONDA_ENV>

# Distributed training configuration
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=8
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Memory and compute optimizations
export CUDA_MODULE_LOADING=EAGER
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# CUDA library paths (adjust for your cluster)
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# HuggingFace cache directory (optional)
export HF_HOME=~/.cache/huggingface

# Navigate to repository
cd <PATH_TO_REPO>

# ===================================================================
# Training Command
# 
# This example shows SFT training. For RL training, replace
# sft_training.py with rl_training.py and adjust arguments.
# ===================================================================

echo "Starting training..."

torchrun --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  sft_training.py \
  --model_name "Qwen/Qwen3-14B" \
  --dataset_path "<PATH_TO_DATASET>" \
  --output_dir "<PATH_TO_OUTPUT>" \
  --block_size 2048 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 20 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True \
  --deepspeed configs/deepspeed_config.json

echo "Training completed!"

# ===================================================================
# Notes:
# 
# 1. Multi-node training: Increase --nodes and adjust --nnodes
# 2. Different GPU count: Adjust --gres=gpu:N and --nproc_per_node=N
# 3. Memory issues: Reduce batch size or increase gradient accumulation
# 4. For RL training: Use rl_training.py and add --sft_checkpoint_path
# ===================================================================
