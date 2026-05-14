#!/bin/bash
# Training script for Hi-MAR on ImageNet
# Reproduces results from Table 2

# Hi-MAR-B (244M parameters)
torchrun --nproc_per_node=8 train.py \
    --task imagenet \
    --model hi_mar_b \
    --data_path /path/to/imagenet \
    --vae_path pretrained/kl16.ckpt \
    --img_size 256 \
    --low_res_img_size 128 \
    --vae_stride 16 \
    --epochs 800 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 0.02 \
    --warmup_epochs 100 \
    --beta1 0.9 \
    --beta2 0.95 \
    --mask_ratio_min 0.7 \
    --mask_ratio_max 1.0 \
    --num_sampling_steps 100 \
    --class_dropout_prob 0.1 \
    --use_ema \
    --ema_momentum 0.9999 \
    --output_dir output/hi_mar_b_imagenet \
    --save_freq 50 \
    --fp16
