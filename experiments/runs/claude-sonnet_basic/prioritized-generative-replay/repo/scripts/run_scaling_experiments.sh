#!/bin/bash
# Run scaling experiments from Section 5.3 / Figure 7 of the paper.
# Tests: (a) larger networks, (b) higher synthetic ratio, (c) higher UTD

ENV="quadruped-walk"
SEEDS=(0 1 2)
DEVICE=${DEVICE:-"cuda"}

echo "=== Scaling Experiment (a): Larger Networks ==="
for SEED in "${SEEDS[@]}"; do
    # PGR with larger network
    python train.py \
        --env "$ENV" \
        --mode pgr \
        --relevance curiosity \
        --seed "$SEED" \
        --device "$DEVICE" \
        --large_network \
        --total_steps 100000 \
        --save_dir results/scaling_network

    # SYNTHER with larger network
    python train.py \
        --env "$ENV" \
        --mode synther \
        --seed "$SEED" \
        --device "$DEVICE" \
        --large_network \
        --total_steps 100000 \
        --save_dir results/scaling_network
done

echo "=== Scaling Experiment (b): Higher Synthetic Ratio ==="
for SEED in "${SEEDS[@]}"; do
    # PGR with r=0.75
    python train.py \
        --env "$ENV" \
        --mode pgr \
        --relevance curiosity \
        --seed "$SEED" \
        --device "$DEVICE" \
        --high_synthetic_ratio \
        --total_steps 100000 \
        --save_dir results/scaling_ratio

    # SYNTHER with r=0.75
    python train.py \
        --env "$ENV" \
        --mode synther \
        --seed "$SEED" \
        --device "$DEVICE" \
        --high_synthetic_ratio \
        --total_steps 100000 \
        --save_dir results/scaling_ratio
done

echo "=== Scaling Experiment (c): Higher UTD (Combined) ==="
for SEED in "${SEEDS[@]}"; do
    # PGR with larger network + r=0.75 + UTD=40
    python train.py \
        --env "$ENV" \
        --mode pgr \
        --relevance curiosity \
        --seed "$SEED" \
        --device "$DEVICE" \
        --large_network \
        --high_synthetic_ratio \
        --high_utd \
        --total_steps 100000 \
        --save_dir results/scaling_combined

    # SYNTHER with larger network + r=0.75 + UTD=40
    python train.py \
        --env "$ENV" \
        --mode synther \
        --seed "$SEED" \
        --device "$DEVICE" \
        --large_network \
        --high_synthetic_ratio \
        --high_utd \
        --total_steps 100000 \
        --save_dir results/scaling_combined
done

echo "All scaling experiments complete!"
