#!/bin/bash
# Stage 1b Pre-training: Train on 185M high-quality data
# Additionally unfreeze text attention parameters
# Global batch size: 7000

set -e

MODEL_NAME="NaViL-2B"
OUTPUT_DIR="./outputs/navil_2b_stage1b"
DATA_PATH="./data/pretrain_stage1b.json"
RESUME_FROM="./outputs/navil_2b_stage1a/checkpoint_final"

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29501}

GLOBAL_BATCH_SIZE=7000
PER_DEVICE_BATCH_SIZE=8
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NNODES * NPROC_PER_NODE)))

echo "Training NaViL Stage 1b"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train.py \
    --stage pretrain_1b \
    --model_name ${MODEL_NAME} \
    --resume_from ${RESUME_FROM} \
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
    --use_multiscale true \
    --logging_steps 100 \
    --save_steps 5000 \
    --num_epochs 1
