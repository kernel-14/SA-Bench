# Baselines for SemanticAlign-Bench

`baselines/` 放的是我们用来生成候选 repo 的基线系统。SemanticAlign-Bench 的测试流程不是“看代码能不能跑”，而是先让系统根据论文生成仓库，再把生成结果交给 `sa_bench`，按 SAU claims 逐条评分。

## 系统概览

| 系统 | 作用 | 输入 | 备注 |
| --- | --- | --- | --- |
| `BasicAgent` | 我们自己的 ReAct baseline | `data/papers/<paper_id>/paper.md` + `config.yaml` + `blacklist.txt` | 静态-only，工具是 `bash` / `read_file_chunk` / `search_file` / `submit` |
| `PaperCoder` | 专门的 paper-to-code 多 agent 系统 | PDF -> S2ORC JSON | 三阶段：planning / analysis / generation |
| `OpenHands` | 通用 coding agent scaffold | 论文文本或 markdown | 批量测试时用 headless/static 模式 |

## 我们怎么用它测试自己的 benchmark

1. 准备输入数据
   - 每个 paper 放在 `data/papers/<paper_id>/`
   - 关键文件是 `paper.md`、`config.yaml`、`sau.json`、`blacklist.txt`
   - 其中 `sau.json` 是 `sa_bench` 的评分依据

2. 生成候选 repo
   - `BasicAgent`：用 `experiments/run_basic_batch.py` 批量跑
   - `OpenHands`：用 `experiments/run_openhands_batch.py` 批量跑
   - `PaperCoder`：先按它自己的 README 把 PDF 转成 JSON，再运行它的脚本
   - 最终用于评分的目录约定是 `experiments/runs/<generator>/<paper>/repo`

3. 评分
   - 批量评分：`python experiments/run_sau_score.py`
   - 单篇评分：`python -m sa_bench --paper <paper_id> --repo <path>/repo --data-dir data/papers`
   - `run_sau_score.py` 会把结果写到 `repo/result/<paper>-sau-score.json`
   - 批量评分默认读取 `.env` 里的 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL=https://api.deepseek.com`

## 常用命令

BasicAgent:

```bash
PAPERBENCH_RUNS_ROOT=experiments/runs/gpt4o_basic \
python experiments/run_basic_batch.py \
  --model gpt-4o \
  --concurrency 4
```

OpenHands:

```bash
PAPERBENCH_RUNS_ROOT=experiments/runs/openhands_deepseek \
python experiments/run_openhands_batch.py \
  --model deepseek-v4-pro \
  --api-base https://api.deepseek.com \
  --concurrency 2
```

PaperCoder:

- 详细流程看 [`PaperCoder/README.md`](PaperCoder/README.md)

## 目录约定

- `data/papers/`：benchmark 输入
- `experiments/runs/`：生成出来的 repo 和评分结果
- `experiments/specs/`：每次运行的配置快照
- `experiments/logs/`：generator 日志
- `experiments/score_progress/`：批量评分进度

## 相关入口

- [`../README.md`](../README.md)：项目总说明
- [`../experiments/run_sau_score.py`](../experiments/run_sau_score.py)：批量 SAU 评分
- [`../sa_bench/cli.py`](../sa_bench/cli.py)：单篇或自定义 repo 的评分 CLI
