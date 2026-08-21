---
language:
- en
license: cc-by-4.0
pretty_name: SemanticAlign-Bench
homepage: https://github.com/kernel-14/SemanticAlign-Bench
repository: https://github.com/kernel-14/SemanticAlign-Bench
tags:
- research-papers
- machine-learning
- benchmarking
- llm-evaluation
- factuality
- structured-extraction
- paper-understanding
- paper-to-code
task_categories:
- question-answering
- text-generation
size_categories:
- 1K<n<10K
annotations_creators:
- expert-generated
language_creators:
- found
---

# SemanticAlign-Bench

A benchmark for evaluating AI agents on **structured claim extraction** from top-tier ML conference papers. Each paper is decomposed into Semantic Alignment Units (SAU) — atomic, self-contained implementation propositions — across four diagnostic dimensions spanning numerical precision to pipeline-level workflow. Agents are evaluated on whether they can reproduce these claims without hallucination, omission, or misordering.

## Dataset Description

- **Papers**: 30 papers from **ICLR 2025**, **ICML 2025**, and **NeurIPS 2025**, spanning 5 domains (6 papers each):

  | Domain | Count |
  |---|---|
  | Probabilistic Inference / Generative Models | 6 |
  | Reinforcement Learning | 6 |
  | Computer Vision | 6 |
  | NLP / LLM | 6 |
  | Numerical Methods / Scientific Computing | 6 |

- **Total SAU Claims**: **1,491**
- **Size**: ~519 MB

### The Four SAU Dimensions

Each paper is decomposed into claims across four diagnostic dimensions, ordered from micro to macro:

| Dimension | Name | Count | Definition |
|-----------|------|-------|------------|
| **D1** | Numerical Precision | 523 | Hyperparameters, configuration values, thresholds, scaling factors |
| **D2** | Formulas / Algorithms | 503 | Mathematical formulas, algorithm steps, architectural mechanisms |
| **D3** | Experiment Protocols | 300 | Datasets, baselines, evaluation metrics, experimental scope |
| **D4** | Pipelines / Procedures | 165 | Multi-step execution order: phase ordering, algorithm step sequencing |

The D1--D4 hierarchy is universal across all evaluated configurations: D1 > D2 > D4 > D3 in score holds invariant for all 12 generator setups (Claude/DeepSeek/Gemini/GPT-4o × BasicAgent/PaperCoder/OpenHands). D3 (experimental protocol) is the dominant bottleneck, with only 0.7% perfect-score rate — 14× lower than D1. D4 exhibits a distinctive pattern: lowest zero rate (33.7%) but only 5.9% of claims score ≥0.5, meaning agents almost always attempt ordering constraints but rarely get them right.

### Paper Venue Distribution

| Venue | Count |
|-------|-------|
| ICLR 2025 | 15 |
| ICML 2025 | 8 |
| NeurIPS 2025 | 7 |

## Dataset Structure

### Per-Paper Directory Layout

```
<paper_id>/
  config.yaml       # Paper metadata (title, venue, year, domain, arxiv URL)
  paper.md          # Full paper text in markdown
  paper.pdf         # Original PDF
  sau.json          # SAU claims — the core annotation file
  images/           # Paper figures extracted from PDF
  blacklist.txt     # Tokens excluded from extraction (e.g., author names)
```

### SAU Claim Format (`sau.json`)

```json
{
  "paper_id": "adjoint-matching",
  "paper_title": "Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC",
  "D1": [
    {
      "id": "adjoint-matching-D1-001",
      "claim": "Image resolution for autoencoder pre-training and generation: 512×512",
      "source": "Section 7"
    }
  ],
  "D2": [ ... ],
  "D3": [ ... ],
  "D4": [ ... ]
}
```

Each claim includes:
- `id`: Unique identifier (`{paper}-{dimension}-{number}`)
- `claim`: Self-contained implementation proposition in natural language
- `source`: Paper section where the claim originates

### Annotation Quality

All 1,491 claims have undergone **multi-version human review** (v4--v5) with systematic error checks:
- Verification against source paper for factual accuracy
- Format normalization and consistency validation
- Cross-reference integrity checks between dimensions
- Fairness audit across domains and paper types (theory vs. empirical)

## Supported Tasks

1. **Claim-Level Factuality**: Given a paper, can the agent accurately extract a specific numerical value, formula, experimental detail, or procedural step?
2. **Dimension-Level Completeness**: Can the agent achieve full recall across all four SAU dimensions for a given paper?
3. **Cross-Dimensional Consistency**: Are claims in D4 (pipelines) consistent with D2 (formulas) and D3 (experiments)?
4. **Hallucination Detection**: Can the agent distinguish paper-supported claims from plausible but fabricated ones?

## Dataset Creation

### Source Data

30 papers selected from ICLR 2025, ICML 2025, and NeurIPS 2025, covering 5 domains with equal representation across task types (classification, generation, RL, theory, scientific computing).

### Annotation Process

**Extraction pipeline** — a multi-agent architecture with 3 specialized extraction agents, each spawning sub-agents per 1--2 sections to avoid attention degradation:

- **Agent 1 (Numerical Auditor, targeting D1):** Uses regex-based extraction with code-first hard-filter rules — strips citations, theorem numbers, and natural-language quantities. Retains only configurable values. Parent agent deduplicates across sections.
- **Agent 2 (Method Parser, targeting D2):** Extracts all code-implementable method components with complete variable definitions. Records `ordering_before`/`ordering_after` annotations for downstream D4 derivation.
- **Agent 3 (Protocol Enumerator, targeting D3):** Extracts four-element experimental protocols: what is compared, on what data, against what baselines, with what metrics. Includes defensive merge validation — if ≥30% of sub-agent outputs fail validation, re-spawns with reduced scope.

**D4 Derivation:** D4 claims are derived from ordering annotations in D2/D3 during the merge phase. Only explicitly stated constraints qualify (numbered Step 1→2→3, Phase 1→2→3 training pipelines, "X before Y" statements). Naturally implied orders and code-engineering call chains are excluded.

**Human Verification:** Each paper undergoes 1--2 hours of expert review across 4--5 iterations (v1→v5), handling ambiguous cases around ablation hyperparameters and derivation/implementation boundary formulas.

### Cost

Multi-agent parallel extraction + 1--2h human review per paper.

## Evaluation Results

In a benchmark study evaluating 360 paper-level runs (12 generators × 30 papers):

- **Overall SAS**: mean 0.221, median 0.237. 82.4% of SAU claims score ≤0.25.
- **Model dominance**: Model choice drives 2.35× more score variation than scaffold choice (1.15×). Top 5 configurations all use Claude or DeepSeek; bottom 3 all use GPT-4o.
- **Scaffold asymmetry**: PaperCoder (+0.116 for GPT-4o) provides more benefit to weaker models. OpenHands adds near-zero value without minimum planning competence.
- **Failure pattern**: Of 7,034 zero-scored claims, 40.8% are implementation mismatches, 16.2% are stubs/placeholders, and 8.0% stem from external knowledge gaps. Improving scores requires better comprehension, not broader coverage.
- **Paper difficulty**: Numerical methods/PDE papers dominate the easiest tier; multi-modal systems and complex training pipelines the hardest.

## Considerations for Using the Data

### Limitations

This is a **static benchmark**: claims test specification fidelity (did the agent encode the right parameters, formulas, and protocols?) rather than runtime correctness. The benchmark does not include execution-based evaluation or dynamic testing.

### Intended Use

- Benchmarking LLM factuality on scientific content
- Measuring agent understanding of structured paper content
- Stress-testing retrieval-augmented generation (RAG) over academic papers

### Out-of-Scope Uses

- Training data for production LLMs (limited size, single annotator)
- Automated paper review or acceptance prediction

## Additional Information

### License

SAU annotations are licensed under **CC-BY-4.0**. Underlying papers are subject to their original copyright terms as posted on arXiv and respective conference proceedings.

### Papers List

| Paper ID | Title | Venue |
|----------|-------|-------|
| adjoint-matching | Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC | ICLR 2025 |
| avg-reward-pg | Global Convergence of Policy Gradient in Average Reward MDPs | ICLR 2025 |
| ca2-vdm | Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing | ICML 2025 |
| cara | Canonical Rank Adaptation: An Efficient Fine-Tuning Strategy for Vision Transformers | ICML 2025 |
| conformal-bayesian-quadrature | Conformal Prediction as Bayesian Quadrature | ICML 2025 |
| diffusion-convergence-rate | Instance-dependent Convergence Theory for Diffusion Models | ICLR 2025 |
| emergent-planning-rl | Interpreting Emergent Planning in Model-Free RL | ICLR 2025 |
| gated-attention-llm | Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free | NeurIPS 2025 |
| generator-augmented-flows | Improving Consistency Models with Generator-Augmented Flows | ICML 2025 |
| hi-mar | Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots | ICML 2025 |
| lora-sb | Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning | ICLR 2025 |
| luno | Linearization Turns Neural Operators into Function-Valued Gaussian Processes | ICML 2025 |
| ma-rlhf | MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions | ICLR 2025 |
| masked-diffusion-token-ordering | Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions | ICML 2025 |
| moe-pot | Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training | NeurIPS 2025 |
| mrq | Towards General-Purpose Model-Free RL (MR.Q) | ICLR 2025 |
| navil | NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints | NeurIPS 2025 |
| neural-operator-flow-matching-pde | Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model | NeurIPS 2025 |
| nfig | NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering | NeurIPS 2025 |
| ngpt | nGPT: Normalized Transformer with Representation Learning on the Hypersphere | ICLR 2025 |
| olmoe | OLMoE: Open Mixture-of-Experts Language Models | ICLR 2025 |
| prioritized-generative-replay | Prioritized Generative Replay | ICLR 2025 |
| pyramidal-flow-matching | Pyramidal Flow Matching for Efficient Video Generative Modeling | ICLR 2025 |
| robotic-world-model | Robotic World Model: A Neural Network Simulator for Robust Policy Optimization | NeurIPS 2025 |
| sam2 | SAM 2: Segment Anything in Images and Videos | ICLR 2025 |
| sc-fno | Sensitivity-Constrained Fourier Neural Operators (SC-FNO) | ICLR 2025 |
| score | Training Language Models to Self-Correct via Reinforcement Learning | ICLR 2025 |
| universal-neural-operators | Towards Universal Neural Operators through Multiphysics Pretraining | NeurIPS 2025 |
| voting-leaderboards | Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards | ICML 2025 |
| wdno | Wavelet Diffusion Neural Operator (WDNO) | ICLR 2025 |
