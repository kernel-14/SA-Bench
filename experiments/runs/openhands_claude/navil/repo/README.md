# NaViL: Native Multimodal Large Language Model

Reproduction of **NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints**.

NaViL is a native MLLM that jointly optimizes vision and language in an end-to-end manner using next-token prediction. Key contributions:
- Modality-specific MoE (both attention and FFN experts) for handling heterogeneous visual/linguistic data
- Optimal visual encoder scaling: encoder size scales log-proportionally with LLM size
- Visual Multi-scale Packing for any-resolution image understanding
- Two-stage training recipe achieving competitive performance with ~600M pre-training pairs

## Repository Structure

```
repo/
├── config.py        # All model and training hyperparameters
├── layers.py        # Primitive layers: RoPE, attention, FFN, MoE experts
├── modules.py       # Visual encoder, MoE-extended LLM, connector
├── model.py         # Full NaViL model with multi-scale packing
├── data.py          # Dataset loading and preprocessing
├── train.py         # Training loop (Stage 1 pre-training + Stage 2 SFT)
├── evaluate.py      # Evaluation on 14 multimodal benchmarks
├── requirements.txt
└── README.md
```

## Model Variants

| Model     | Visual Encoder | LLM (activated) | Total A-Params | LLM Base       |
|-----------|---------------|-----------------|----------------|----------------|
| NaViL-2B  | 0.6B (d=24, w=1472) | 1.8B (d=24, w=2048) | 2.4B | InternLM2-1.8B |
| NaViL-9B  | 1.2B (d=32, w=1792) | 8.0B (d=36, w=4096) | 9.2B | Qwen3-8B       |

## Architecture

- **Visual Encoder**: Transformer with bidirectional attention and 2D-RoPE, same layer structure as LLM
- **Connector**: Pixel shuffle downsampling + MLP projection to LLM hidden dim
- **LLM**: Causal transformer with 1D-RoPE and modality-specific MoE
- **MoE**: Separate Q/K/V/O projections and FFN (gate/up/down) per modality (visual vs linguistic)
- **Special tokens**: `<begin_of_image>`, `<end_of_image>`, `<end_of_line>`, `<end_of_scale>`

## Training Stages

**Stage 1.1** (500M image-text pairs): Freeze LLM text params; train visual encoder, MLP projector, MoE visual experts.

**Stage 1.2** (185M high-quality data): Unfreeze LLM attention text params for cross-modal alignment.

**Stage 2** (68M high-quality data): Unfreeze all params for supervised fine-tuning.

## Usage

```bash
# Train NaViL-2B
python train.py --config navil_2b --stage 1

# Evaluate on benchmarks
python evaluate.py --model_path /path/to/checkpoint --benchmarks mmvet mmmu mmbench
```
