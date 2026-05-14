# NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints

This repository reproduces the core contributions of the NaViL paper:

> **NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints**  
> Changyao Tian, Hao Li, Gen Luo, Xizhou Zhu, et al.  
> Shanghai AI Laboratory, CUHK, Tsinghua University, Sensetime Research, Nanjing University

## Overview

NaViL is a **native** Multimodal Large Language Model (MLLM) that jointly optimizes vision and language in an end-to-end manner using Next-Token-Prediction (NTP). Unlike compositional MLLMs (e.g., LLaVA, InternVL) that use separately pre-trained visual encoders, NaViL trains all components together.

### Key Contributions

1. **Architecture Design Principles** (Section 3.2):
   - **Observation 1**: Initializing from pre-trained LLM greatly benefits convergence (10x faster than training from scratch)
   - **Observation 2**: Modality-specific MoEs (MHA-MMoE + FFN-MMoE) significantly improve performance without increasing inference cost
   - **Observation 3**: Visual encoders achieve near-optimal performance across a wide range of depth/width configurations

2. **Scaling Properties** (Section 3.3):
   - **Observation 4**: Scaling LLM follows conventional scaling laws; scaling visual encoder shows diminishing returns
   - **Observation 5**: Optimal visual encoder size scales log-proportionally with LLM size (unlike compositional MLLMs that use fixed encoder size)

3. **NaViL Model** (Section 4):
   - NaViL-2B: InternLM2-1.8B + 600M visual encoder, 2.4B activated params
   - NaViL-9B: Qwen3-8B + 1.2B visual encoder, 9.2B activated params
   - Visual Multi-scale Packing for any-resolution input
   - Competitive with top-tier compositional MLLMs on 14 benchmarks

## Repository Structure

```
navil/
├── navil/
│   ├── __init__.py          # Package exports
│   ├── model.py             # NaViL main model (NaViLModel, NaViLConfig)
│   ├── visual_encoder.py    # Visual encoder V_{d,w}(I) with 2D-RoPE
│   ├── moe.py               # Modality-specific MoE (MHA-MMoE + FFN-MMoE)
│   ├── data.py              # Data processing, datasets, image preprocessing
│   ├── trainer.py           # Training utilities and NaViLTrainer
│   └── scaling_analysis.py  # Scaling experiment analysis tools
├── configs/
│   ├── navil_2b.yaml        # NaViL-2B configuration
│   └── navil_9b.yaml        # NaViL-9B configuration
├── scripts/
│   ├── train.py             # Main training script
│   ├── train_pretrain_stage1a.sh  # Stage 1a training script
│   ├── train_pretrain_stage1b.sh  # Stage 1b training script
│   └── train_sft.sh         # Stage 2 SFT script
├── evaluation/
│   └── evaluate.py          # Evaluation on 14 benchmarks
├── tools/
│   └── run_scaling_experiments.py  # Scaling experiment runner
└── requirements.txt
```

## Architecture Details

### Visual Encoder V_{d,w}(I)

The visual encoder consists of `d` transformer layers with hidden dimension `w`:

```
V_{d,w}(I) = C ⊙ F_d^w ⊙ ... ⊙ F_1^w ⊙ P(I)
```

- **P**: Patch Embedding (stride=16, pads image to multiples of 32)
- **F_i^w**: Transformer layer with **bidirectional attention** and **2D-RoPE**
- **C**: Connector (pixel shuffle downsampling + MLP projection to LLM space)
- Parameter count: N ≈ 12 × d × w²

For NaViL-2B: d=24, w=1472 → ~624M parameters

### Modality-specific MoE (MMoE)

Each LLM layer uses modality-specific experts for both attention and FFN:

**MHA-MMoE** (modality-specific Q, K, V, O projections):
```
MHA-MMoE(x_{i,m}) = (softmax(QK^T/√d) V) W_O^m
Q_{i,m} = x_{i,m} W_Q^m,  K_{i,m} = x_{i,m} W_K^m,  V_{i,m} = x_{i,m} W_V^m
```

**FFN-MMoE** (modality-specific gate, up, down projections):
```
FFN-MMoE(x_{i,m}) = (SiLU(x_{i,m} W_gate^m) ⊙ x_{i,m} W_up^m) W_down^m
```

where m ∈ {visual, linguistic}. Each token activates exactly 1 expert (its modality), maintaining consistent inference cost.

### Visual Multi-scale Packing

For any-resolution input with downsampling rate τ = √2/2:
- I_0: original image
- I_1: I_0 × τ
- I_2: I_1 × τ
- ... until area < threshold

Special tokens: `<begin_of_image>`, `<end_of_image>`, `<end_of_line>`, `<end_of_scale>`

## Training Recipe

### Stage 1: Multi-modal Generative Pre-training

**Stage 1a** (500M samples, global batch=7000):
- Data: 300M web-scale (Laion-2B, Coyo-700M, Wukong, SA-1B) + 200M synthetic captions
- Frozen: all text parameters
- Trainable: visual encoder, MLP connector, MoE visual experts (index 0)
- Multi-scale: disabled

**Stage 1b** (185M samples, global batch=7000):
- Data: high-quality multimodal + pure language
- Additionally unfreeze: text attention parameters (Q/K/V/O projections)
- Multi-scale: enabled

### Stage 2: Supervised Fine-tuning

**SFT** (68M samples, global batch=4614):
- All parameters unfrozen
- High-quality multimodal instruction data
- Multi-scale: enabled

## Scaling Findings

### Optimal Encoder Size Scaling (Key Finding)

The paper shows that the optimal visual encoder size scales log-proportionally with LLM size:

```
log(optimal_encoder_size) = α × log(llm_size) + β
```

This means compositional MLLMs (which use a fixed encoder size across all LLM scales) are suboptimal.

| LLM Size | Optimal Encoder Size |
|----------|---------------------|
| 0.5B     | ~75-150M            |
| 1.8B     | ~300-600M           |
| 7B       | ~600M-1.2B          |

### Optimal Encoder Size Definition

The smallest encoder whose loss difference compared to an encoder twice its size is less than λ = 1% of the loss with the 75M encoder.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training NaViL-2B

```bash
# Stage 1a: Pre-training (visual components only)
bash scripts/train_pretrain_stage1a.sh

# Stage 1b: Pre-training (+ text attention)
bash scripts/train_pretrain_stage1b.sh

# Stage 2: SFT
bash scripts/train_sft.sh
```

### Evaluation

```bash
python evaluation/evaluate.py \
    --checkpoint ./outputs/navil_2b_sft/checkpoint_final/model.pt \
    --benchmark TextVQA \
    --data_path ./data/textvqa_val.json
```

### Scaling Analysis

```bash
# Run all scaling experiments
python tools/run_scaling_experiments.py --experiment all

# Analyze results
python tools/run_scaling_experiments.py --experiment analyze
```

## Benchmark Results

### MLLM Benchmarks (Table 1)

| Model | #A-Param | Avg | MMVet | MMMU | MMBench | MME | MathVista | OCRBench | CCBench |
|-------|----------|-----|-------|------|---------|-----|-----------|----------|---------|
| NaViL-2B | 2.4B | **67.1** | **78.3** | 41.8 | 71.2 | 1822 | 50.0 | 796 | **83.9** |
| Mono-InternVL | 1.8B | 56.4 | 40.1 | 33.7 | 65.5 | 1875 | 45.7 | 767 | 66.3 |
| EVEv2 | 7B | 53.2 | 45.0 | 39.3 | 66.3 | 1709 | 60.0 | 702 | 30.8 |

### VQA Benchmarks (Table 2)

| Model | #A-Param | Avg | TextVQA | SQA-I | GQA | DocVQA | AI2D | ChartQA | InfoVQA |
|-------|----------|-----|---------|-------|-----|--------|------|---------|---------|
| NaViL-2B | 2.4B | **75.1** | 76.9 | 95.0 | 59.8 | 85.4 | 74.6 | 78.0 | 56.0 |
| Mono-InternVL | 1.8B | 70.1 | 72.6 | 93.6 | 59.5 | 80.0 | 68.6 | 73.7 | 43.0 |

## Assumptions and Unresolved Details

1. **LLM initialization**: The paper uses InternLM2-Base for experiments. The exact weight mapping from InternLM2 to NaViL's MoE-extended architecture requires careful implementation. Our `_load_llm_weights` function provides a template but may need adjustment for specific model versions.

2. **MoE layer selection**: The paper states "all LLM layers use MMoE" but doesn't specify if this applies to all 24 layers or a subset. We implement all layers using MMoE.

3. **Connector architecture**: The paper mentions "pixel shuffle and MLP projector" but doesn't specify the exact MLP depth. We use a 2-layer MLP with GELU activation.

4. **Training data**: The exact data mixtures and filtering criteria for the 185M high-quality data and 68M SFT data are not fully specified. We provide the framework but not the actual data.

5. **Attention map visualization** (Fig. 9): The attention visualization code is not included as it requires a trained model checkpoint.

6. **NaViL-9B details**: The appendix mentions NaViL-9B uses Qwen3-8B with 1.2B visual encoder and disables multi-scale packing in stage 1a for acceleration. We implement this in `configs/navil_9b.yaml`.

7. **Validation loss metric**: The paper uses teacher-forcing loss on a held-out subset of the multimodal dataset. Our implementation uses standard cross-entropy loss.

## Citation

```bibtex
@article{tian2025navil,
  title={NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints},
  author={Tian, Changyao and Li, Hao and Luo, Gen and Zhu, Xizhou and Su, Weijie and Deng, Hanming and Zhu, Jinguo and Shao, Jie and Zhu, Ziran and Liu, Yunpeng and Lu, Lewei and Wang, Wenhai and Li, Hongsheng and Dai, Jifeng},
  year={2025}
}
```
