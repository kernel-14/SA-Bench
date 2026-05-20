# SemanticAlign-Bench

Benchmark for evaluating semantic alignment in LLM-based paper reproduction. Measures whether AI agents faithfully reproduce ML papers — not just whether the code runs, but whether it encodes the right parameters, formulas, experimental protocols, and procedural steps.

**Paper**: [Anonymous submission]  
**Dataset**: [HuggingFace](https://anonymous-hf.up.railway.app/a/rrgn430zpfui/)  
**Benchmark**: 30 papers × 1,491 SAU claims across ICLR/ICML/NeurIPS 2025

---

## Quick Start

### 1. Download the Dataset

```bash
huggingface-cli download [anonymous]/SemanticAlign-Bench --local-dir ./data
```

The dataset contains 30 papers with SAU (Semantic Alignment Unit) claims in four diagnostic dimensions:

| Dimension | Type | Count | Example |
|-----------|------|-------|---------|
| D1 | Numerical Precision | 523 | Learning rate, batch size, thresholds |
| D2 | Formulas / Algorithms | 503 | Loss functions, attention mechanisms |
| D3 | Experiment Protocols | 300 | Datasets, baselines, metrics |
| D4 | Pipelines / Procedures | 165 | Training phase ordering, algorithm steps |

### 2. Run Scoring

```bash
# Score one paper
python -m sa_bench \
  --paper adjoint-matching \
  --repo /path/to/generated/repo \
  --dim D1 \
  --model deepseek-v4-pro \
  --api-key $OPENAI_API_KEY \
  --base-url $OPENAI_BASE_URL

# Score all papers (batch)
python -m sa_bench \
  --paper all \
  --repo-root /path/to/generated_repos/ \
  --data-dir data/papers \
  --model deepseek-v4-pro \
  --output results.json \
  --workers 4
```

**CLI Arguments**:

| Flag | Description |
|------|-------------|
| `--paper` | Paper ID or `all` |
| `--dim` | Dimension: `D1`/`D2`/`D3`/`D4` or `all` (default) |
| `--repo` | Path to single generated repository |
| `--repo-root` | Root dir containing one repo per paper (for `--paper all`) |
| `--data-dir` | Directory with `{paper_id}/sau.json` files (default: `data/papers`) |
| `--model` | Judge model name (default: `deepseek-v4-pro`) |
| `--api-key` | OpenAI-compatible API key (or set `OPENAI_API_KEY`) |
| `--base-url` | OpenAI-compatible base URL (or set `OPENAI_BASE_URL`) |
| `--output` | Write JSON result to file |
| `--workers` | Parallel workers for batch scoring |
| `--llm-search` | Use LLM ReAct search (default: deterministic grep-based) |
| `--no-llm-judge` | Heuristic scoring fallback instead of LLM judge |

### 3. Run Baselines

Three scaffold systems in `baselines/`:

**BasicAgent** — our ReAct baseline
```bash
cd baselines/BasicAgent
python run.py --paper data/papers/adjoint-matching/paper.md --output ./output/
```

**PaperCoder** — specialized paper-to-code agent ([source](https://github.com/HimJoe/paper2code))

**OpenHands** — general coding scaffold ([source](https://github.com/All-Hands-AI/OpenHands))

Setup instructions for PaperCoder and OpenHands in `baselines/README.md`.

---

## Evaluation Results

360 evaluations across 12 generator configurations (4 models × 3 scaffolds × 30 papers). Results in `experiments/runs/`.

| Scaffold | Claude Sonnet | DeepSeek | Gemini Flash | GPT-4o |
|----------|--------------|----------|-------------|--------|
| BasicAgent | 0.272 | 0.268 | 0.150 | 0.081 |
| PaperCoder | 0.301 | 0.241 | 0.256 | 0.198 |
| OpenHands | 0.277 | 0.282 | 0.243 | 0.082 |

**Key findings**:
- Overall SAU score mean: 0.221, median: 0.237. 82.4% of claims score ≤0.25.
- Model effect (2.35× range) dominates scaffold effect (1.15× range)
- D1 > D2 > D4 > D3 hierarchy invariant across all 12 configs
- D3 (experimental protocol) is the bottleneck: 0.7% perfect-score rate, 14× lower than D1
- Zero-scored claims (7,034 total): 40.8% implementation mismatch, 16.2% stubs/placeholders, 8.0% external knowledge gaps

### Dimension Statistics

| Statistic | D1 | D2 | D3 | D4 |
|-----------|----|----|----|-----|
| Mean | 0.290 | 0.244 | 0.160 | 0.190 |
| Median | 0.304 | 0.235 | 0.167 | 0.188 |
| % ≥0.5 | 28.7% | 18.2% | 3.7% | 5.9% |
| Full-mark rate (1.0) | 9.7% | 5.4% | 0.7% | 0.9% |
| Zero rate | 39.6% | 35.8% | 44.8% | 33.7% |

### Marginal Means by Model and Scaffold

| Factor | Overall | D1 | D2 | D3 | D4 |
|--------|---------|----|----|----|-----|
| **By Model** | | | | | |
| Claude-Sonnet-4.6 | 0.283 | 0.413 | 0.293 | 0.201 | 0.225 |
| DeepSeek-V4-Pro | 0.263 | 0.358 | 0.294 | 0.186 | 0.216 |
| Gemini-2.5-Flash | 0.217 | 0.268 | 0.248 | 0.161 | 0.189 |
| GPT-4o | 0.120 | 0.120 | 0.140 | 0.092 | 0.130 |
| **By Scaffold** | | | | | |
| PaperCoder | 0.249 | 0.344 | 0.262 | 0.179 | 0.211 |
| OpenHands | 0.221 | 0.288 | 0.233 | 0.174 | 0.188 |
| BasicAgent | 0.193 | 0.238 | 0.237 | 0.127 | 0.170 |

### Full Generator × Scaffold Scores

| Scaffold | Model | Overall | D1 | D2 | D3 | D4 |
|----------|-------|---------|----|----|----|-----|
| BasicAgent | Claude-Sonnet-4.6 | 0.272 | 0.391 | 0.307 | 0.169 | 0.219 |
| BasicAgent | DeepSeek-V4-Pro | 0.268 | 0.379 | 0.308 | 0.180 | 0.204 |
| BasicAgent | Gemini-2.5-Flash | 0.150 | 0.134 | 0.210 | 0.099 | 0.158 |
| BasicAgent | GPT-4o | 0.081 | 0.047 | 0.121 | 0.058 | 0.100 |
| PaperCoder | Claude-Sonnet-4.6 | 0.301 | 0.470 | 0.299 | 0.209 | 0.226 |
| PaperCoder | DeepSeek-V4-Pro | 0.241 | 0.339 | 0.278 | 0.148 | 0.200 |
| PaperCoder | Gemini-2.5-Flash | 0.256 | 0.335 | 0.270 | 0.198 | 0.222 |
| PaperCoder | GPT-4o | 0.198 | 0.231 | 0.200 | 0.163 | 0.197 |
| OpenHands | Claude-Sonnet-4.6 | 0.276 | 0.379 | 0.272 | 0.224 | 0.229 |
| OpenHands | DeepSeek-V4-Pro | 0.282 | 0.357 | 0.297 | 0.229 | 0.244 |
| OpenHands | Gemini-2.5-Flash | 0.243 | 0.335 | 0.264 | 0.187 | 0.187 |
| OpenHands | GPT-4o | 0.082 | 0.082 | 0.098 | 0.056 | 0.093 |
| **Mean** | | **0.221** | **0.290** | **0.244** | **0.160** | **0.190** |

### Per-Paradigm Scores

| Paradigm | Papers | Overall | D1 | D2 | D3 | D4 |
|----------|--------|---------|----|----|----|-----|
| New Algorithm / Architecture | 13 | 0.232 | 0.307 | 0.269 | 0.164 | 0.188 |
| Theoretical Analysis | 5 | 0.225 | 0.276 | 0.222 | 0.192 | 0.211 |
| System / Pipeline | 3 | 0.217 | 0.269 | 0.231 | 0.163 | 0.203 |
| Empirical Comparison | 2 | 0.212 | 0.239 | 0.290 | 0.145 | 0.173 |
| Generative Models | 2 | 0.210 | 0.351 | 0.190 | 0.125 | 0.172 |
| Incremental Improvement | 5 | 0.198 | 0.268 | 0.210 | 0.135 | 0.179 |

---

## PaperBench Pilot Study

Claude Sonnet 4.6 + BasicAgent (ReAct, max\_steps=80, time\_limit=900s) on five ICML 2024 papers from PaperBench-dev, evaluated by the official GPT-4o judge pipeline (`code_only=True`). Raw resource usage harvested from each run's `meta.json` and `grade.json`.

| Paper | Score | Pass | Total | Steps | Time (s) | Cost | Termination |
|-------|-------|------|-------|-------|----------|------|-------------|
| mechanistic-understanding | 0.839 | 29 | 36 | 37 | 1252.5 | $5.81 | Time limit |
| sample-specific-masks | 0.810 | 66 | 87 | 81 | 652.4 | $6.10 | Step limit |
| pinn | 0.806 | 112 | 126 | 35 | 909.4 | $4.10 | Time limit |
| fre | 0.331 | 113 | 306 | 24 | 909.1 | $2.11 | Time limit |
| all-in-one | 0.117 | 15 | 92 | 81 | 478.1 | $3.41 | Step limit |

The D1--D4 drift taxonomy was derived inductively from 312 score-0 leaf nodes across these five runs (see paper Appendix for judge reasoning excerpts and failure analysis).

---

## Repository Structure

```
SemanticAlign-Bench/
├── sa_bench/                  # SAU scoring pipeline (Python SDK + CLI)
│   ├── cli.py                 #   CLI entry point
│   ├── scorer.py              #   SAUScorer public API
│   ├── pipeline.py            #   Scoring orchestration
│   ├── judge.py               #   Dimension-specific LLM judge
│   ├── index.py               #   RepoIndex for code search
│   └── prompts/               #   D1-D4 judge system prompts
├── baselines/                 # Agent scaffolds
│   ├── BasicAgent/            #   Our ReAct baseline
│   ├── PaperCoder/            #   Specialized paper-to-code
│   └── OpenHands/             #   General coding scaffold
├── experiments/
│   ├── runs/                  #   Per-config results (30 papers × 12 configs)
│   └── specs/                 #   Experiment configuration files
├── paperbench_case_study/     # Pilot study: Sonnet 4.6 on 5 ICML 2024 papers
│   ├── papers/                #   Source PDFs
│   └── sonnet46-basicagent/   #   Generated repos + analysis
```

## Citation

```bibtex
@inproceedings{semanticalign_bench,
  title     = {SemanticAlign-Bench: Evaluating Semantic Alignment in LLM-Based Paper Reproduction},
  author    = {Anonymous Author(s)},
  year      = {2025},
  note      = {Dataset at \url{https://anonymous-hf.up.railway.app/a/rrgn430zpfui/}}
}
```

## License

Code: MIT. Dataset annotations: CC-BY-4.0.
