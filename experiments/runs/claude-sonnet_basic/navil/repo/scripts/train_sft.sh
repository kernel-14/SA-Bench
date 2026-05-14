#!/bin/bash
# Stage 2: Supervised Fine-tuning on 68M high-quality multimodal data
# All parameters unfrozen
# Global batch size: 4614

set -e

MODEL_NAME="NaViL-2B"
OUTPUT_DIR="./outputs/navil_2b_sft"
DATA_PATH="./data/sft_data.json"
RESUME_FROM="./outputs/navil_2b_stage1b/checkpoint_final"

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29502}

# Global batch size = 4614 (from paper)
GLOBAL_BATCH_SIZE=4614
PER_DEVICE_BATCH_SIZE=4
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NNODES * NPROC_PER_NODE)))

echo "Training NaViL Stage 2 (SFT)"
echo "  Global batch size: ${GLOBAL_BATCH_SIZE}"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train.py \
    --stage sft \
    --model_name ${MODEL_NAME} \
    --resume_from ${RESUME_FROM} \
    --output_dir ${OUTPUT_DIR} \
    --data_path ${DATA_PATH} \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 2e-5 \
    --min_lr 2e-6 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --bf16 \
    --max_seq_len 8192 \
    --image_size 448 \
    --use_multiscale true \
    --logging_steps 50 \
    --save_steps 1000 \
    --num_epochs 1
