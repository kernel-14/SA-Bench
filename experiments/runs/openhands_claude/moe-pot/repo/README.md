# MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training

Implementation of the MoE-POT architecture from the paper:
> "Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training"
> Hong Wang et al., USTC

## Repository Structure

```
repo/
├── config.py       Model configurations (Tiny/Small/Medium) and hyperparameters
├── layers.py       Primitive layers: ConvExpert, RouterGating, FourierHead,
│                   PatchEmbedding, TemporalAggregation
├── modules.py      Composed modules: FourierLayer, MoELayer, MoEBlock,
│                   OutputProjection
├── model.py        Full MoE-POT model with auto-regressive forward pass
├── data.py         Dataset loading, preprocessing, balanced sampling,
│                   noise injection
├── train.py        Pre-training, fine-tuning, and downstream task training
├── evaluate.py     Zero-shot evaluation, router interpretability analysis,
│                   rollout error, expert usage ratio
├── utils.py        AverageMeter, L2RE metric, checkpoint I/O
└── requirements.txt
```

## Architecture

MoE-POT processes spatiotemporal PDE data through four stages:

1. **Patchification** — Conv2D with stride P=8 maps each input frame to patch tokens with learnable positional encodings `p_{i,j}^t = W_p(x_i, y_j, t)`.

2. **Temporal Aggregation** — Fourier-weighted sum across T=10 frames:
   `z_agg = Σ_t W_t(z_p^t) * e^{-iγt}`

3. **N MoE Blocks** — Each block contains:
   - **Fourier Layer**: multi-head frequency-domain MLP `F^{-1}[W2·σ(W1·F[z]+b1)+b2]`
   - **MoE Layer**: 2 shared experts (always active) + top-4 of 16 routed experts selected by a CNN router

4. **Output Projection** — Transposed convolution upsamples back to original resolution.

### MoE Layer Output

```
z^{l+1} = (1/N_s) Σ_i shared_i(z) + Σ_k w_k · routed_{i_k}(z)
```

### Load Balancing Loss

```
L_balance = w_bal · CV({Importance_i})²
Importance_i = Σ_b w_{i,b}
```

## Model Sizes

| Size   | Attn dim | MLP dim | Layers | Heads | Total params | Activated |
|--------|----------|---------|--------|-------|-------------|-----------|
| Tiny   | 512      | 512     | 4      | 4     | ~30M        | ~17M      |
| Small  | 1024     | 1024    | 6      | 8     | ~166M       | ~90M      |
| Medium | 1024     | 2048    | 8      | 8     | ~489M       | ~288M     |

## Training

### Pre-training (1000 epochs, 6 datasets)

```bash
torchrun --nproc_per_node=8 train.py \
    --mode pretrain \
    --model_size tiny \
    --data_root /path/to/data \
    --batch_size 20
```

### Fine-tuning (200 epochs, router frozen)

```bash
python train.py \
    --mode finetune \
    --model_size small \
    --data_root /path/to/data \
    --dataset fno_ns_1e5 \
    --pretrain_ckpt checkpoints/pretrain_small_epoch1000.pt
```

### Downstream Tasks (500 epochs)

```bash
python train.py \
    --mode downstream \
    --model_size small \
    --data_root /path/to/data \
    --dataset pdearena \
    --pretrain_ckpt checkpoints/pretrain_small_epoch1000.pt
```

## Evaluation

```bash
# Zero-shot L2RE on all 6 datasets
python evaluate.py --mode zero_shot --model_size small \
    --checkpoint checkpoints/pretrain_small_epoch1000.pt \
    --data_root /path/to/data

# Router classification accuracy (Section 5.4)
python evaluate.py --mode classify --model_size small \
    --checkpoint checkpoints/pretrain_small_epoch1000.pt \
    --data_root /path/to/data --block_idx 1

# Expert usage ratio per dataset (Figure 2 right)
python evaluate.py --mode usage --model_size small \
    --checkpoint checkpoints/pretrain_small_epoch1000.pt \
    --data_root /path/to/data --block_idx 3

# Rollout error at frames 50/70/100 (Appendix C.3)
python evaluate.py --mode rollout --model_size small \
    --checkpoint checkpoints/pretrain_small_epoch1000.pt \
    --data_root /path/to/data --datasets pdebench_swe
```

## Datasets

Pre-training uses 6 datasets from 3 benchmarks:

| Name              | Source     | PDE type                        | Train  | Test  |
|-------------------|------------|---------------------------------|--------|-------|
| fno_ns_1e5        | FNO        | Navier-Stokes (ν=1e-5)          | 1000   | 200   |
| fno_ns_1e3        | FNO        | Navier-Stokes (ν=1e-3)          | 1000   | 200   |
| pdebench_cns      | PDEBench   | Compressible NS (η=0.1, ζ=0.01) | 9000   | 200   |
| pdebench_swe      | PDEBench   | Shallow Water Equations         | 900    | 60    |
| pdebench_dr       | PDEBench   | Diffusion-Reaction              | 900    | 60    |
| cfdbench          | CFDBench   | Incompressible NS (irregular)   | 9000   | 1000  |

Data files should be placed at `<data_root>/<dataset_name>/{train,test}.npy` with shape `(N, T, C, H, W)`.

## Key Hyperparameters

| Parameter          | Value  | Description                          |
|--------------------|--------|--------------------------------------|
| T                  | 10     | Input timesteps                      |
| P (patch size)     | 8      | Spatial patch size                   |
| H (spatial size)   | 128    | Standardized spatial resolution      |
| N_r                | 16     | Number of routed experts             |
| N_s                | 2      | Number of shared experts             |
| K (top-K)          | 4      | Experts selected per forward pass    |
| w_bal              | 0.1    | Load balancing loss weight           |
| lr                 | 1e-3   | Learning rate (Adam)                 |
| β₁, β₂             | 0.9    | Adam momentum parameters             |
| weight_decay       | 1e-6   | L2 regularization                    |
| ε (noise scale)    | 0.01   | Noise injection scale (pre-training) |
