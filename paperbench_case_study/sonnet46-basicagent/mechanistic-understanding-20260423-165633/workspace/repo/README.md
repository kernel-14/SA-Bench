# A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity

This repository reproduces the paper "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity" (2024).

## Overview

The paper investigates *how* DPO (Direct Preference Optimization) reduces toxicity in GPT2-medium, using mechanistic interpretability tools. The key findings are:

1. **Toxic MLP value vectors**: GPT2 contains MLP value vectors that encode toxic content (identified via cosine similarity with a linear toxicity probe W_toxic).
2. **DPO mechanism**: DPO reduces toxicity by learning an offset in the residual stream that avoids activating toxic MLP value vectors (specifically, it shifts the residual stream away from the activation regions gamma(MLP.k_Toxic)).
3. **Un-alignment**: The alignment can be reversed by scaling the toxic key vectors by 10x, which expands the activation regions and causes the residual stream to pass through them again.

## Repository Structure

```
src/
  train_probe.py          - Train linear toxicity probe on Jigsaw dataset
  extract_toxic_vectors.py - Extract toxic MLP value vectors from GPT2
  generate_pplm_dataset.py - Generate pairwise toxic/nontoxic dataset using PPLM
  train_dpo.py            - Train GPT2 with DPO on the pairwise dataset
  evaluate.py             - Evaluate toxicity, perplexity, and F1
  analyze_dpo_mechanism.py - Analyze the DPO mechanism (Figures 2, 3, 5)
  visualize.py            - Generate paper figures (Figures 1, 4, 5)
  unalign.py              - Un-align GPT2_DPO by scaling toxic key vectors
data/                     - Generated datasets
checkpoints/              - Saved model checkpoints
results/                  - Evaluation results and figures
```

## Pipeline

### Step 1: Train Toxicity Probe

Train a linear probe on the Jigsaw dataset to identify the toxic direction W_toxic in GPT2's residual stream:

```bash
python src/train_probe.py \
  --output_dir checkpoints/probe \
  --batch_size 32 \
  --epochs 5 \
  --lr 1e-3
```

The probe achieves ~96% accuracy on the Jigsaw validation set. The toxic direction vector W_toxic[:, 1] is saved as `checkpoints/probe/W_toxic.npy`.

### Step 2: Extract Toxic Value Vectors

Find MLP value vectors with highest cosine similarity to W_toxic:

```bash
python src/extract_toxic_vectors.py \
  --w_toxic_path checkpoints/probe/W_toxic.npy \
  --output_dir checkpoints/toxic_vectors \
  --top_n 128
```

This identifies the top-128 toxic value vectors and applies SVD to find the principal toxic directions (SVD.U_toxic).

### Step 3: Generate PPLM Dataset

Generate 24,576 pairwise (toxic, nontoxic) continuations using PPLM:

```bash
python src/generate_pplm_dataset.py \
  --w_toxic_path checkpoints/probe/W_toxic.npy \
  --output_dir data/pplm_pairs \
  --num_pairs 24576 \
  --stepsize 0.4 \
  --num_iterations 50 \
  --gm_scale 0.95 \
  --kl_scale 0.1 \
  --top_k 10
```

- Positive (chosen) samples: greedy decoding from GPT2
- Negative (rejected) samples: PPLM-guided generation using W_toxic as attribute classifier

### Step 4: Train DPO

Train GPT2-medium with DPO on the pairwise dataset:

```bash
python src/train_dpo.py \
  --train_data data/pplm_pairs/train_pairs.json \
  --val_data data/pplm_pairs/val_pairs.json \
  --output_dir checkpoints/dpo \
  --lr 1e-6 \
  --batch_size 4 \
  --beta 0.1 \
  --patience 10
```

DPO hyperparameters (from Table 8):
- Learning rate: 1e-6
- Batch size: 4
- Optimizer: RMSProp
- Max gradient norm: 10
- DPO beta: 0.1
- Validation patience: 10

### Step 5: Evaluate

Evaluate GPT2 and GPT2_DPO on toxicity, perplexity, and F1:

```bash
# Evaluate GPT2 (baseline)
python src/evaluate.py \
  --model_path gpt2-medium \
  --output_dir results/gpt2

# Evaluate GPT2_DPO
python src/evaluate.py \
  --model_path checkpoints/dpo/gpt2_dpo_best \
  --output_dir results/gpt2_dpo

# Evaluate with SUBTRACT intervention (W_toxic subtraction)
python src/evaluate.py \
  --model_path gpt2-medium \
  --subtract_vector checkpoints/probe/W_toxic.npy \
  --alpha 1.0 \
  --output_dir results/gpt2_subtract
```

Expected results (from Table 1):
| Model | Toxicity | Perplexity | F1 |
|-------|----------|------------|-----|
| GPT2 | 0.527 | 35.6 | 0.29 |
| GPT2_DPO | 0.288 | 37.2 | 0.28 |
| GPT2_SUBTRACT | 0.288 | 37.2 | 0.28 |

### Step 6: Analyze DPO Mechanism

Analyze how DPO reduces toxicity:

```bash
python src/analyze_dpo_mechanism.py \
  --dpo_model_path checkpoints/dpo/gpt2_dpo_best \
  --toxic_vectors_info checkpoints/toxic_vectors/toxic_vectors_info.json \
  --output_dir results/analysis
```

This generates:
- `activation_results.json`: Mean activations of toxic vectors before/after DPO
- `mean_delta_x_layer19.npy`: Mean residual stream shift at layer 19
- `figure2_mean_activations.png`: Figure 2 from the paper
- `figure5_cos_sim.png`: Figure 5 from the paper

### Step 7: Generate Figures

```bash
python src/visualize.py \
  --dpo_model_path checkpoints/dpo/gpt2_dpo_best \
  --output_dir results/figures
```

Generates:
- `figure1_logit_lens.png`: Logit lens showing P("shit") across layers
- `figure4_pca_shift.png`: PCA plot of residual stream shift
- `figure5_cos_sim.png`: Cosine similarity distribution

### Step 8: Un-alignment

Reverse the alignment by scaling toxic key vectors:

```bash
python src/unalign.py \
  --dpo_model_path checkpoints/dpo/gpt2_dpo_best \
  --w_toxic_path checkpoints/probe/W_toxic.npy \
  --output_dir checkpoints/unaligned \
  --top_k 7 \
  --scale_factor 10.0
```

Then evaluate the un-aligned model:

```bash
python src/evaluate.py \
  --model_path checkpoints/unaligned/gpt2_dpo_unaligned \
  --output_dir results/gpt2_dpo_unaligned
```

Expected: toxicity returns to ~GPT2 baseline levels.

## Key Findings

### Mechanistic Explanation

DPO reduces toxicity through the following mechanism:

1. **Toxic value vectors**: GPT2 contains MLP value vectors v_i^l that encode toxic content. These are identified by high cosine similarity with W_toxic (the toxic direction from the linear probe).

2. **Activation regions**: Each toxic value vector v_i^l is activated when the residual stream x^{l-mid} falls in the region gamma(k_i^l) = {x : sigma(x . k_i^l) > 0}.

3. **DPO offset**: DPO learns to shift the residual stream x^{l-mid} by a constant offset delta_x that moves it out of the activation regions gamma(MLP.k_Toxic). This is achieved by modifying the MLP value vectors in earlier layers.

4. **Linear shift**: The shift is approximately linear (constant across prompts), as shown by the PCA plot (Figure 4).

5. **Un-alignment**: Scaling the toxic key vectors k_i^l by 10x expands the activation regions, causing the residual stream to pass through them again, reverting the model to toxic behavior.

### Key Equations

The MLP output at layer l is:
```
MLP^l(x) = sum_i sigma(x . k_i^l) * v_i^l
```

where k_i^l are key vectors (rows of c_fc.weight) and v_i^l are value vectors (columns of c_proj.weight).

DPO learns an offset delta_x such that:
```
x_DPO^{l-mid} = x_GPT2^{l-mid} + delta_x
```

where delta_x is approximately constant across prompts and moves the residual stream out of the toxic activation regions.

## Assumptions and Unresolved Details

1. **Jigsaw dataset**: The paper uses the Jigsaw Toxic Comment Classification Challenge dataset. We use the HuggingFace version `thesofakillers/jigsaw-toxic-comment-classification-challenge`.

2. **PPLM dataset generation**: The paper uses PPLM with W_toxic as the attribute classifier. The exact PPLM hyperparameters are from Table 9 (stepsize=0.4, num_iterations=50, gm_scale=0.95, kl_scale=0.1, top_k=10).

3. **Toxic vector selection**: The paper selects the top-128 value vectors by cosine similarity with W_toxic, then applies SVD. The exact threshold for "toxic" vs "non-toxic" is not specified.

4. **Layer 19**: The paper focuses on layer 19 as the most important layer for toxicity. This is identified empirically via the logit lens analysis.

5. **Un-alignment**: The paper uses top-7 toxic key vectors scaled by 10x. The exact selection criterion is not fully specified.

## References

- Rafailov et al. (2023): Direct Preference Optimization
- Dathathri et al. (2019): Plug and Play Language Models (PPLM)
- Geva et al. (2022): Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space
- Gehman et al. (2020): RealToxicityPrompts
