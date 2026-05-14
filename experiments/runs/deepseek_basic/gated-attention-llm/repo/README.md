# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

Reproduction of the paper: "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
by Qiu et al. (Qwen Team, Alibaba Group).

## Paper Summary

This work systematically investigates gating mechanisms in standard softmax attention. The central finding: applying a **head-specific sigmoid gate after Scaled Dot-Product Attention (SDPA)** consistently improves performance, training stability, and scaling properties. The paper explores 30+ gating variants across 15B MoE and 1.7B dense models.

### Key Findings Reproduced in this Codebase:

1. **Gating Positions**: G₁ (after SDPA) > G₂ (after Value) > G₃/G₄/G₅ (Table 1)
2. **Gating Granularity**: Head-specific > Head-shared (Table 1, rows 10-13)
3. **Gating Mode**: Multiplicative > Additive (Table 1, row 14 vs 5)
4. **Activation**: Sigmoid > SiLU > Identity (Table 1, row 5 vs 15)
5. **Non-linearity mechanism**: Gating introduces non-linearity between W_V and W_O low-rank mappings (Sec 4.1, Table 3)
6. **Sparsity mechanism**: SDPA output gate introduces query-dependent sparsity (Sec 4.2, Table 4)
7. **Attention-Sink elimination**: Gated models show no attention sink (Fig. 2, Sec 4.3)
8. **Training stability**: Gating prevents loss spikes, enables larger learning rates (Table 2, Fig. 1)
9. **Long-context extrapolation**: Gated models maintain performance at 128k context (Table 5, Sec 4.4)

## Repository Structure

```
gated_attention/
├── __init__.py
├── modules/
│   ├── __init__.py
│   ├── gating.py              # Core gated attention implementation
│   └── transformer_block.py   # Decoder blocks with gated attention and FFN (dense + MoE)
├── models/
│   ├── __init__.py
│   └── gated_llm.py           # Full model (15A2B MoE, 1.7B dense) with configs
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training loop, LR scheduler, stability monitoring
│   └── data.py                # Data loading utilities
├── analysis/
│   ├── __init__.py
│   └── attention_analysis.py  # Attention sink, sparsity, gate score analysis
├── configs/
│   ├── __init__.py
│   └── paper_configs.py       # All paper experiment configurations (Tables 1-5)
├── utils/
│   ├── __init__.py
│   └── evaluation.py          # PPL, benchmark evaluation, RULER
└── tests/
    ├── __init__.py
    └── test_gating.py         # Tests for all gating variants
```

## Installation

```bash
pip install -e .
```

Requirements:
- Python >= 3.8
- PyTorch >= 1.12.0

## Usage

### Quick Demo
```bash
python run_experiments.py --mode demo
```

### List All Paper Variants
```bash
python run_experiments.py --list_variants
```

### Build a Specific Model
```bash
# Build SDPA-gated MoE model (Table 1, row 5)
python run_experiments.py --mode build --model_type 15A2B --gating_variant g1_elementwise

# Build baseline
python run_experiments.py --mode build --model_type 15A2B --gating_variant baseline

# Build dense model
python run_experiments.py --mode build --model_type 1.7B_28L --gating_variant g1_elementwise
```

### Build All Table 1 Variants
```bash
python run_experiments.py --mode build_table --table table1
```

### Run Analysis
```bash
python run_experiments.py --mode analyze --model_type 15A2B --gating_variant g1_elementwise --seq_len 256
```

### Programmatic Usage
```python
from gated_attention.models.gated_llm import create_model_from_paper_config
from gated_attention.analysis.attention_analysis import AttentionAnalyzer

# Create model
model = create_model_from_paper_config("15A2B", "g1_elementwise")

# Analyze attention patterns
analyzer = AttentionAnalyzer(model)
analysis = analyzer.analyze(input_ids)

print(f"Attention sink ratio: {analysis['average_sink_ratio']:.4f}")
```

## Gating Variants

### Positions (Fig. 1)
| Position | Description | Key |
|----------|-------------|-----|
| G₁ | After SDPA output | Most effective |
| G₂ | After Value projection | Second best |
| G₃ | After Key projection | Minimal improvement |
| G₄ | After Query projection | Minimal improvement |
| G₅ | After Dense output | No effect |

### Configuration Options
- **Granularity**: `elementwise` (per-dimension) or `headwise` (per-head scalar)
- **Scope**: `head_specific` or `head_shared`
- **Mode**: `multiplicative` (Y' = Y · σ(XW)) or `additive` (Y' = Y + σ(XW))
- **Activation**: `sigmoid`, `silu`, `ns_sigmoid`, `identity`

## Model Architectures

### 15A2B MoE (Table 1)
- 48 layers, d_model=2048, 32 query heads, 4 KV heads, head_dim=128
- 128 fine-grained experts with top-8 softmax routing
- 15B total parameters, 2.54B activated
- GQA (Grouped Query Attention)
- Z-loss for load balancing

### 1.7B Dense (Table 2)
- Configurable: 28 or 48 layers
- d_model=2048 (28L) or d_model=1536 (48L)
- 16 query heads, 4 KV heads
- SwiGLU FFN (with reduced width when using gating to match params)
- Optional sandwich normalization

## Training Configurations

| Setting | LR | Batch Size | Tokens |
|---------|-----|-----------|--------|
| MoE 15A2B (400B) | 2e-3 | 1024 | 400B |
| Dense 1.7B (400B) | 4e-3 | 1024 | 400B |
| Dense 1.7B (3.5T) | 4.5e-3 | 2048 | 3.5T |
| Dense 1.7B (1T) | 5.3e-3 | 4096 | 1T |
| High LR experiments | 8e-3 | varies | varies |

All use cosine decay to 3e-5 with 1000-step warmup.

## Key Mechanisms Explained

### Non-linearity (Sec 4.1)
In multi-head attention, W_V and W_O form a low-rank linear mapping (rank = d_k < d_model). Adding gating at G₁ or G₂ introduces non-linearity between these projections, increasing expressiveness (Equations 6-8).

### Sparsity (Sec 4.2)
Effective gating scores are sparse (most values near 0). SDPA output gating scores exhibit the strongest sparsity, and this sparsity is:
- **Query-dependent**: SDPA gating depends on the current query state
- **Head-specific**: Different heads require different sparsity patterns
- The NS-sigmoid variant (scores ∈ [0.5, 1.0]) removes sparsity while preserving non-linearity, proving sparsity is necessary for full gains (Table 4, row 7)

### Attention-Sink Elimination (Sec 4.3)
- Baseline models: ~46.7% attention to first token
- Gated models: ~4.8% attention to first token
- Value gating (G₂) reduces massive activations but NOT attention sinks
- Head-shared gating or NS-sigmoid restores both massive activations and attention sinks
- Conclusion: sparse, query-dependent, head-specific gating is required

## Assumptions and Missing Details

1. **Tokenizer**: The paper uses a Qwen-family tokenizer (vocab_size=151936). The exact tokenizer is not specified in detail.

2. **Exact FFN dimensions**: For dense models, the paper mentions reducing FFN width to match parameter counts when using gating. The exact reduction ratio needs calibration.

3. **Data mixture**: The 3.5T token dataset composition (multilingual, math, general knowledge) proportions are not specified numerically.

4. **RoPE implementation**: Standard RoPE with base=10000 is used. For long-context experiments, base is increased to 1M and YaRN is applied, but exact YaRN parameters are not detailed.

5. **Sandwich norm placement**: Applied to attention/FFN outputs before residual addition, as described in Ding et al. (2021).

6. **Switch Head experiments** (Appendix A.1): The equivalence "Switch v, 1top1 = v Headwise Gate (Table 1, row 11)" is noted.

7. **Benchmark evaluation**: Few-shot settings are standard (Hellaswag 10-shot, MMLU 5-shot, GSM8k 5-shot, HumanEval 0-shot, C-eval 5-shot, CMMLU 5-shot). Integration with lm-evaluation-harness would be needed for exact reproduction.

## Testing

```bash
cd gated_attention/tests
python test_gating.py
```

Tests cover:
- All 5 gating positions (G1-G5)
- Elementwise vs headwise granularity
- Multiplicative vs additive modes
- Head-specific vs head-shared scopes
- All activation functions (sigmoid, SiLU, NS-sigmoid, identity)
- All 11 paper variants from Table 1
- Attention sink ratio computation
- Gate score statistics computation

## References

- Vaswani et al., 2017: "Attention Is All You Need"
- Xiao et al., 2023: "Efficient Streaming Language Models with Attention Sinks"
- Shazeer, 2020: "GLU Variants Improve Transformer"
- Ainslie et al., 2023: "GQA: Training Generalized Multi-Query Transformer Models"
- Dai et al., 2024: "DeepSeekMoE: Towards Ultimate Expert Specialization"
- Su et al., 2024: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Peng et al., 2023: "YaRN: Efficient Context Window Extension"
- Hsieh et al., 2024: "RULER: What's the Real Context Size?"
- Ding et al., 2021: "CogView: Mastering Text-to-Image Generation via Transformers"
- Zhang & Sennrich, 2019: "Root Mean Square Layer Normalization"
