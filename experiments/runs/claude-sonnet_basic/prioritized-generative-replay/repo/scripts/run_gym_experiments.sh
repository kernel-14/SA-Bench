#!/bin/bash
# Run OpenAI Gym experiments from Table 2 of the paper.
# Each experiment is run with 3 seeds.

ENVS=("Walker2d-v2" "HalfCheetah-v2" "Hopper-v2")
SEEDS=(0 1 2)
DEVICE=${DEVICE:-"cuda"}

# PGR (Curiosity)
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running PGR (Curiosity) on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode pgr \
            --relevance curiosity \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/gym_100k
    done
done

# SYNTHER baseline
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running SYNTHER on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode synther \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/gym_100k
    done
done

# REDQ baseline
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running REDQ on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode redq \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/gym_100k
    done
done

echo "All Gym experiments complete!"
