#!/bin/bash
# Train SCoRe on MBPP dataset with Gemini 1.0 Pro settings
# Hyperparameters from Table 5 (right): MBPP

python -m score.train \
    --task code \
    --model_name_or_path "google/gemma-2-9b-it" \
    --base_model_path "google/gemma-2-9b-it" \
    --output_dir ./outputs/score_code \
    --stage1_steps 750 \
    --stage2_steps 750 \
    --total_steps 1500 \
    --batch_size 128 \
    --learning_rate 1e-5 \
    --stage1_beta2 0.25 \
    --stage1_beta1 0.01 \
    --stage2_beta1 0.01 \
    --stage2_alpha 10.0 \
    --max_grad_norm 1.0 \
    --sampling_temperature 1.0 \
    --save_every 500 \
    --eval_every 500
