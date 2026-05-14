#!/bin/bash
# Training script for Hi-MAR on MS-COCO text-to-image generation
# Reproduces results from Table 3

torchrun --nproc_per_node=8 train.py \
    --task coco \
    --model hi_mar_s \
    --data_path /path/to/coco \
    --vae_path pretrained/kl16.ckpt \
    --img_size 256 \
    --low_res_img_size 128 \
    --vae_stride 16 \
    --epochs 400 \
    --batch_size 256 \
    --lr 8e-4 \
    --weight_decay 0.03 \
    --warmup_epochs 10 \
    --beta1 0.9 \
    --beta2 0.95 \
    --num_sampling_steps 100 \
    --use_ema \
    --ema_momentum 0.9999 \
    --output_dir output/hi_mar_s_coco \
    --save_freq 50 \
    --fp16
