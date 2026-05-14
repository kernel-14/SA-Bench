#!/bin/bash
# Stage 1a Pre-training: Train visual components on 500M image-text pairs
# Frozen: text parameters
# Trainable: visual encoder, MLP connector, MoE visual experts
# Global batch size: 7000

set -e

# Configuration
MODEL_NAME="NaViL-2B"
OUTPUT_DIR="./outputs/navil_2b_stage1a"
DATA_PATH="./data/pretrain_stage1a.json"
LLM_INIT="internlm/internlm2-1_8b"  # InternLM2-1.8B base model

# Distributed training settings
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

# Training hyperparameters
# Global batch size = 7000 (from paper)
# Per-device batch size depends on GPU count
GLOBAL_BATCH_SIZE=7000
PER_DEVICE_BATCH_SIZE=8
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NNODES * NPROC_PER_NODE)))

echo "Training NaViL Stage 1a"
echo "  Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "  Per-device batch size: ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM}"
echo "  Nodes: ${NNODES}, GPUs per node: ${NPROC_PER_NODE}"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train.py \
    --stage pretrain_1a \
    --model_name ${MODEL_NAME} \
    --llm_init ${LLM_INIT} \
    --output_dir ${OUTPUT_DIR} \
    --data_path ${DATA_PATH} \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 1e-4 \
    --min_lr 1e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --bf16 \
    --max_seq_len 4096 \
    --image_size 448 \
    --patch_size 16 \
    --use_multiscale false \
    --logging_steps 100 \
    --save_steps 5000 \
    --num_epochs 1
