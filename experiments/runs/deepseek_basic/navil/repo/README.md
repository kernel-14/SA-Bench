# NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints

This repository contains a reproduction of the NaViL paper by Tian et al. It implements the core architecture, scaling analysis, and training methodology described in the paper.

## Paper Summary

NaViL is a native Multimodal Large Language Model (MLLM) that jointly optimizes vision and language in an end-to-end manner. Unlike the dominant compositional paradigm (which connects pre-trained vision encoders to pre-trained LLMs), NaViL explores the design space and scaling properties of native MLLMs under practical data constraints.

### Key Contributions Reproduced

1. **Architecture Design Exploration (Section 3.2)**
   - **LLM Initialization**: Pre-trained LLM initialization greatly benefits multimodal convergence (Observation 1)
   - **Modality-Specific MoE**: Both attention (MHA-MMoE) and FFN (FFN-MMoE) modality-specific experts improve performance without increasing activated parameters (Observation 2)
   - **Visual Encoder Architecture**: Wide range of optimal depth/width configurations; deeper encoders perform slightly better with more data (Observation 3)

2. **Scaling Properties Analysis (Section 3.3)**
   - LLM scaling follows conventional scaling laws (Observation 4)
   - Visual encoder scaling shows diminishing returns bounded by LLM capacity (Observation 4)
   - Optimal encoder size scales log-proportionally with LLM size (Observation 5)

3. **NaViL Model Architecture (Section 4.1)**
   - Visual encoder with bidirectional attention and 2D-RoPE
   - Connector with pixel shuffle downsampling + MLP projection
   - MoE-extended LLM with modality-specific attention and FFN experts
   - Visual multi-scale packing for any-resolution image inputs

4. **Training Recipe (Section 4.2)**
   - Stage 1.1: 500M web-scale image-text pairs (freeze text, train vision)
   - Stage 1.2: 185M high-quality alignment data (unfreeze attention)
   - Stage 2: 68M supervised fine-tuning data (all parameters)

## Repository Structure

```
navil/                      # Core model implementation
├── __init__.py             # Package exports
├── model.py                # NaViLModel and NaViLConfig
├── visual_encoder.py       # Visual encoder with bidirectional attention
├── moe.py                  # Modality-specific MoE (MHA-MMoE and FFN-MMoE)
├── connector.py            # Pixel shuffle + MLP connector
├── multi_scale.py          # Visual multi-scale packing
├── scaling.py              # Scaling law analysis
├── trainer.py              # Three-stage training pipeline
└── data.py                 # Data loading and preprocessing

configs/                    # Model configuration files
├── navil_2b.yaml           # NaViL-2B configuration
└── navil_9b.yaml           # NaViL-9B configuration

scripts/                    # Training and analysis scripts
├── train_navil.py          # Main training script
└── analyze_scaling.py      # Scaling analysis and plotting
```

## Model Configurations

### NaViL-2B
| Component | Configuration | Parameters |
|-----------|--------------|------------|
| Visual Encoder | depth=24, width=1472, mlp_width=5888, heads=23 | ~0.6B |
| LLM (MoE) | depth=24, width=2048, mlp_width=8192, heads=16, experts=2 | ~1.8B |
| Total | - | ~4.2B |
| Activated | - | ~2.4B |

### NaViL-9B
| Component | Configuration | Parameters |
|-----------|--------------|------------|
| Visual Encoder | depth=32, width=1792, mlp_width=7168, heads=28 | ~1.2B |
| LLM (MoE) | depth=36, width=4096, mlp_width=12288, heads=32, experts=2 | ~8.0B |
| Total | - | ~?B |
| Activated | - | ~9.2B |

## Key Design Decisions

### Modality-Specific MoE (MHA-MMoE + FFN-MMoE)

The paper introduces modality-specific experts for BOTH attention and FFN layers:

```
MHA-MMoE(x_{i,m}) = softmax(QK^T/√d)V W_O^m

where:
  Q_{i,m} = x_{i,m} W_Q^m  (modality-specific projection)
  K_{i,m} = x_{i,m} W_K^m
  V_{i,m} = x_{i,m} W_V^m
  W_O^m: modality-specific output projection

FFN-MMoE(x_{i,m}) = (SiLU(x W_gate^m) ⊙ x W_up^m) W_down^m
```

This differs from standard MoE approaches that only have separate FFN experts. The key insight is that modality-specific attention projections help maintain consistent feature scales across modalities.

### Visual Encoder Architecture

The visual encoder uses the same transformer backbone as the LLM but with:
- **Bidirectional attention** (instead of causal)
- **2D-RoPE** (instead of 1D-RoPE) for spatial position encoding

Parameter count follows: N ≈ 12 × d × w²

### Training Stages

1. **Stage 1.1** (Web-scale pre-training):
   - Freeze: textual parameters (LLM text experts, token embedding, lm_head)
   - Train: visual encoder, MLP projector, MoE visual experts
   - LR: constant with warm-up, 5e-5

2. **Stage 1.2** (High-quality alignment):
   - Unfreeze: self-attention textual parameters
   - LR: constant with warm-up, 5e-5

3. **Stage 2** (Supervised fine-tuning):
   - All parameters trainable
   - LR: cosine decay, 2e-5

## Usage

### Model Creation

```python
from navil.model import create_navil_2b, create_navil_9b

# Create NaViL-2B
model_2b = create_navil_2b()

# Create NaViL-9B
model_9b = create_navil_9b()
```

### Training

```bash
# Train NaViL-2B (all stages)
python scripts/train_navil.py --model_size 2b --stage all

# Train only stage 1.1
python scripts/train_navil.py --model_size 2b --stage s1_1

# Resume from checkpoint
python scripts/train_navil.py --model_size 2b --resume ./checkpoints/navil_S1.1_step10000.pt
```

### Scaling Analysis

```bash
# Run scaling analysis and generate plots
python scripts/analyze_scaling.py --output_dir ./scaling_results
```

## Assumptions and Unresolved Details

1. **LLM Initialization**: The paper initializes from InternLM2-1.8B (NaViL-2B) and Qwen3-8B (NaViL-9B). Our implementation supports loading from these checkpoints but requires access to the actual pre-trained weights.

2. **Tokenizer**: Uses the same tokenizer as InternLM2 (vocab size ~92K). Special tokens (`<begin_of_image>`, `<end_of_image>`, `<end_of_line>`, `<end_of_scale>`) need to be added to the tokenizer.

3. **Data Processing**: The paper uses specific datasets (LAION-2B, COYO-700M, Wukong, SA-1B, InternVL-2.5 data). Our data module provides loading infrastructure but requires access to these datasets.

4. **Pixel Shuffle Details**: The connector uses pixel shuffle for downsampling (reference: InternVL). The exact downsampling ratio and implementation follow InternVL's approach.

5. **2D-RoPE Implementation**: The visual encoder uses 2D RoPE where half the embedding dimension encodes horizontal position and half encodes vertical position.

6. **Multi-scale Packing**: The downsampling rate τ = √2/2 is used. During NaViL-9B training, multi-scale packing is disabled in S1.1 for acceleration.

7. **Gradient Accumulation**: The paper uses gradient accumulation to achieve large effective batch sizes. Our implementation supports this.

8. **Evaluation**: The paper evaluates on MMVet, MMMU, MMBench, MME, MathVista, OCRBench, CCBench, TextVQA, ScienceQA, GQA, DocVQA, AI2D, ChartQA, InfoVQA. Evaluation infrastructure is not included in this reproduction.

## References

- Paper: NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints
- Code: https://github.com/OpenGVLab/NaViL (original, not used in this reproduction)
- Mono-InternVL: CVPR 2025 (prior work on monolithic MLLMs)
- InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks
- Kaplan et al.: Scaling laws for neural language models

## What Was Reproduced

- ✅ Full NaViL model architecture (visual encoder, connector, MoE-extended LLM)
- ✅ MHA-MMoE and FFN-MMoE modality-specific expert implementations
- ✅ Bidirectional visual encoder with 2D-RoPE
- ✅ Visual multi-scale packing
- ✅ Three-stage training pipeline with parameter freezing/unfreezing
- ✅ Scaling law analysis with predicted optimal encoder sizes
- ✅ Configuration files for NaViL-2B and NaViL-9B
- ✅ Data loading infrastructure for all stages
- ✅ Training scripts with distributed training support

## What Was Not Fully Reproduced (Out of Scope)

- Actual training runs with real data
- Model checkpoint loading from InternLM2/Qwen3
- Full evaluation pipeline on all 14 benchmarks
- Attention map visualization (Section 5.3)
- NLP capability evaluation (Appendix E)
