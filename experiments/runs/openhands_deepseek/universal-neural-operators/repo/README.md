# Universal Neural Operators through Multiphysics Pretraining

Reproduction of the paper *"Towards Universal Neural Operators through Multiphysics Pretraining"* (Masliaev et al., 2025).

## Codebase Structure

```
repo/
├── layers.py       # Core building blocks: FNO spectral convolutions, Mamba SSM,
│                   # Perceiver IO, Codomain Attention, SwinV2, Lift/Project adapters
├── models.py       # Full model architectures: FNO, MambaFNO, LocalAttnFNO,
│                   # PerceiverIOFNO, CoDANO, SwinV2FNO, MultiPhysicsNO wrapper
├── data.py         # PDE dataset generators: Burgers, Gray-Scott, Navier-Stokes,
│                   # Heat, Heat+Convection, Reaction-Diffusion+Advection, Advection
├── train.py        # Training loop, NMAE metric (Eq. 3), pretraining, fine-tuning,
│                   # experiment runners for Tables 1 & 2
├── config.py       # All hyperparameters and experiment configurations
├── main.py         # CLI entry point
└── requirements.txt
```

## Implemented Models

| Model | Description | Reference |
|-------|-------------|-----------|
| `FNO` | Baseline Fourier Neural Operator | [Kovachki et al. 2023] |
| `MambaFNO` | FNO + post-lifting Mamba SSM | Section 3, Eq. 2 |
| `LocalAttnFNO` | FNO + post-lifting local attention | Section 3 |
| `PerceiverIOFNO` | Perceiver IO-based neural operator | Section 3, [Jaegle et al. 2021] |
| `CoDANO` | Codomain Attention Neural Operator | Section 3, [Rahman et al. 2024] |
| `SwinV2FNO` | Swin Transformer V2 neural operator | Section 2/3, [Herde et al. 2024] |

## Key Techniques

- **Adapters**: Problem-specific lift/projection layers that map between input/output spaces and the shared hidden representation
- **Pretraining**: Multi-physics training where the core operator body learns shared dynamics across different PDEs
- **Fine-tuning**: Only adapter parameters are trained on a new problem, while the core body is frozen (similar to LoRA)

## Usage

### Single-problem training from scratch
```bash
python main.py --mode train_single --problem burgers --model mamba_fno --epochs 100
```

### Run Table 1 experiments (out-of-sample parameters)
```bash
python main.py --mode table1
```

### Run Table 2 experiments (input extension & multi-physics transfer)
```bash
python main.py --mode table2
```

### Custom configuration
```python
from config import get_config
from train import train

config = get_config('table1')
results = train(config)
```

## Dependencies

```
torch>=2.0.0
numpy
scipy
tqdm
pyyaml
einops
```

## Metric

Range-Normalized Mean Absolute Error (NMAE), as defined in Equation 3:

$$\text{NMAE} = \frac{1}{|\mathcal{D}_{test}|} \sum_{(a,u) \in \mathcal{D}_{test}} \frac{|\mathcal{G}_\theta(a) - u|_{1,G}}{\max_G u - \min_G u + \varepsilon}$$
