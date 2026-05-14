#!/bin/bash
# Run all DMC state-based experiments from Table 1 of the paper.
# Each experiment is run with 5 seeds.

ENVS=("quadruped-walk" "cheetah-run" "reacher-hard" "finger-turn-hard")
SEEDS=(0 1 2 3 4)
DEVICE=${DEVICE:-"cuda"}

# PGR (Curiosity) - main method
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
            --save_dir results/dmc_100k
    done
done

# PGR (Return)
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running PGR (Return) on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode pgr \
            --relevance return \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/dmc_100k
    done
done

# PGR (TD Error)
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running PGR (TD Error) on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode pgr \
            --relevance td_error \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/dmc_100k
    done
done

# PGR (Reward) - naive baseline
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running PGR (Reward) on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode pgr \
            --relevance reward \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/dmc_100k
    done
done

# SYNTHER baseline (unconditional generation)
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running SYNTHER on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode synther \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/dmc_100k
    done
done

# REDQ baseline (model-free)
for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running REDQ on $ENV, seed $SEED"
        python train.py \
            --env "$ENV" \
            --mode redq \
            --seed "$SEED" \
            --device "$DEVICE" \
            --total_steps 100000 \
            --save_dir results/dmc_100k
    done
done

# Finger-turn-hard uses 300K steps (harder sparse reward task)
for SEED in "${SEEDS[@]}"; do
    echo "Running PGR (Curiosity) on finger-turn-hard (300K), seed $SEED"
    python train.py \
        --env finger-turn-hard \
        --mode pgr \
        --relevance curiosity \
        --seed "$SEED" \
        --device "$DEVICE" \
        --total_steps 300000 \
        --save_dir results/dmc_300k
done

echo "All DMC experiments complete!"
