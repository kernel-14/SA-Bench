#!/bin/bash
# Evaluation script for Hi-MAR
# Generates 50K images and computes FID/IS metrics

MODEL=${1:-hi_mar_b}
CHECKPOINT=${2:-output/hi_mar_b_imagenet/checkpoint_epoch0799.pth}
OUTPUT_DIR=${3:-generated/hi_mar_b}
REF_PATH=${4:-/path/to/imagenet/val}

# Generate images with CFG
python generate.py \
    --task imagenet \
    --model ${MODEL} \
    --checkpoint ${CHECKPOINT} \
    --num_samples 50000 \
    --batch_size 128 \
    --cfg_scale 1.5 \
    --num_steps_phase1 32 \
    --num_steps_phase2 4 \
    --output_dir ${OUTPUT_DIR}_cfg \
    --save_images \
    --ref_path ${REF_PATH}

# Generate images without CFG (phase 2 only)
python generate.py \
    --task imagenet \
    --model ${MODEL} \
    --checkpoint ${CHECKPOINT} \
    --num_samples 50000 \
    --batch_size 128 \
    --cfg_scale 1.0 \
    --num_steps_phase1 32 \
    --num_steps_phase2 4 \
    --output_dir ${OUTPUT_DIR}_no_cfg \
    --save_images \
    --ref_path ${REF_PATH}
