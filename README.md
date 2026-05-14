# SemanticAlign-Bench

Benchmark for evaluating semantic alignment in LLM-based paper reproduction. Measures whether AI agents faithfully reproduce ML papers — not just whether the code runs, but whether it encodes the right parameters, formulas, experimental protocols, and procedural steps.

**Paper**: [Anonymous submission]  
**Dataset**: [HuggingFace](https://anonymous-hf.up.railway.app/a/rrgn430zpfui/)  
**Benchmark**: 30 papers × 1,491 SAU claims across ICLR/ICML/NeurIPS 2025

---

## Quick Start

### 1. Download the Dataset

```bash
huggingface-cli download kernel14/SemanticAlign-Bench --local-dir ./data
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
- Overall SAU score mean: 0.221, median: 0.200. 82.4% of claims score ≤0.25.
- Model effect (2.35× range) dominates scaffold effect (1.15× range)
- D1 > D2 > D4 > D3 hierarchy invariant across all 12 configs
- D3 (experimental protocol) is the bottleneck: 0.7% perfect-score rate, 14× lower than D1
- 81% of zero-scored claims contain partial but incorrect code; only 5.7% are completely absent

---

## PaperBench Case Study

`paperbench_case_study/` contains results from a pilot study using Claude Sonnet 4.6 + BasicAgent on five ICML 2024 papers evaluated with PaperBench-dev.

| Paper | Score | Key Drift |
|-------|-------|-----------|
| mechanistic-understanding | 0.839 | Light D1 (eval metrics) |
| sample-specific-masks | 0.810 | Light D1+D2 (baseline gaps) |
| pinn | 0.806 | Light D3 (experiment configs) |
| fre | 0.331 | Heavy D2 (experiments missing, time-truncated) |
| all-in-one | 0.117 | Severe D1+D2+D3 (core architecture missing) |

Full analysis: `paperbench_case_study/sonnet46-basicagent/case_study.md`

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
