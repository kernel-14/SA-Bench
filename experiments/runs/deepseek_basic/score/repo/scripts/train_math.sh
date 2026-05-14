#!/bin/bash
# Train SCoRe on MATH dataset with Gemini 1.5 Flash settings
# Hyperparameters from Table 5 (left): MATH

python -m score.train \
    --task math \
    --model_name_or_path "google/gemma-2-9b-it" \
    --base_model_path "google/gemma-2-9b-it" \
    --output_dir ./outputs/score_math \
    --stage1_steps 1500 \
    --stage2_steps 1500 \
    --total_steps 3000 \
    --batch_size 512 \
    --learning_rate 5e-6 \
    --stage1_beta2 0.1 \
    --stage1_beta1 0.01 \
    --stage2_beta1 0.01 \
    --stage2_alpha 10.0 \
    --max_grad_norm 1.0 \
    --sampling_temperature 1.0 \
    --save_every 500 \
    --eval_every 500
