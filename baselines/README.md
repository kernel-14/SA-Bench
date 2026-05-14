# Baselines for SemanticAlign-Bench

> 目录：`Research_space/baselines/`
> 用途：为 SemanticAlign-Bench 提供 paper→repo 的 agent 系统，生成 final repo 供语义对齐评估

---

## 已 Clone 的系统

```
baselines/
├── OpenHands/    ← 通用 agent scaffold，PaperBench 官方 baseline（21.0%）
├── PaperCoder/   ← 专为 paper→repo 设计，PaperBench 45.1%
└── DeepCode/     ← 多 agent 系统，支持 Paper2Code 模式
```

---

## 1. OpenHands

**来源**：[All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) | ICLR 2025

**定位**：通用 AI 软件工程 agent，能执行代码、浏览网页、操作文件系统。PaperBench 最强 agent 使用 Claude 3.5 Sonnet (New) + open-source scaffolding，得分 21.0%（论文原文：arXiv:2504.01848）。具体 scaffold 框架论文摘要未点名，OpenHands 是当时最主流的开源 coding agent，但需读论文方法章节确认。

**安装**：
```bash
cd OpenHands
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

**Headless batch 模式**（适合批量跑 benchmark）：
```bash
openhands --headless \
  --task "Read /workspace/paper.md and generate a complete executable repository reproducing all experiments." \
  --model anthropic/claude-sonnet-4-5 \
  --workspace ./output/paper_name/
```

**批量脚本**：
```bash
for paper in papers/*.md; do
  name=$(basename $paper .md)
  openhands --headless \
    --task "$(cat paper_repro_prompt.txt)

Paper content:
$(cat $paper)" \
    --model anthropic/claude-sonnet-4-5 \
    --workspace ./outputs/$name
done
```

**Paper reproduction prompt 模板**：
```
You are a research engineer. Reproduce the experiments from the following ML paper.
1. Create a complete Python repository (model.py, train.py, data.py, requirements.txt, README.md)
2. Implement all models, datasets, baselines, and training procedures described in the paper
3. Ensure the code is executable and reproduces the main results
4. Write all files to /workspace/reproduction/
```

**与 PaperBench 的关系**：OpenHands 是 PaperBench 的 reference agent，可直接用 PaperBench harness 运行：
```bash
git clone https://github.com/openai/preparedness
cd preparedness/project/paperbench
# 按 README 配置 OpenHands 作为 agent
```

---

## 2. PaperCoder（Paper2Code）

**来源**：[HimJoe/paper2code](https://github.com/HimJoe/paper2code) | arXiv:2504.17192

**定位**：专为 paper→repo 设计的三阶段多 agent 系统：Planning（架构设计）→ Analysis（算法提取）→ Generation（代码生成）。PaperBench 得分 45.1%，是目前最强的专用系统。

**安装**：
```bash
cd PaperCoder
pip install openai
```

**输入格式**：需要先把 PDF 转成 JSON（用 s2orc-doc2json）：
```bash
# 安装 PDF 转换工具
git clone https://github.com/allenai/s2orc-doc2json.git
cd s2orc-doc2json/grobid-0.7.3 && ./gradlew run  # 启动 Grobid 服务

# 转换 PDF
python s2orc-doc2json/doc2json/grobid2json/process_pdf.py \
  -i paper.pdf -t ./temp/ -o ./output/
```

**运行**：
```bash
cd scripts
bash run.sh  # 默认跑 Attention is All You Need 示例
```

**注意**：PaperCoder 依赖 OpenAI API（o3-mini 或 GPT-4o），不支持 Anthropic。

---

## 3. DeepCode

**来源**：[HKUDS/DeepCode](https://github.com/HKUDS/DeepCode) | arXiv:2512.07921

**定位**：HKU 数据智能实验室开发的多 agent 编程系统，支持 Paper2Code、Text2Web、Text2Backend 三种模式。支持 OpenAI-compatible API（可接 Claude、GPT-5 等）。

**安装（本地模式，无 Docker）**：
```bash
cd DeepCode
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python=3.13
source .venv/bin/activate
uv pip install -r requirements.txt

# 配置 API key
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
# 编辑 mcp_agent.secrets.yaml，填入 API key
```

**运行**：
```bash
deepcode --local  # 本地模式（无 Docker）
# 或
deepcode          # Docker 模式（推荐）
```

**Paper2Code 模式**：在 UI 或 CLI 中选择 Paper2Code 任务类型，输入论文 PDF 或 arXiv URL。

---

## 4. Claude Sonnet 直接 API（无 scaffold）

**定位**：最简单的 lower bound baseline，单次生成，无 agent loop，无工具调用。

**运行**：
```python
import anthropic

client = anthropic.Anthropic()
paper_text = open("paper.md").read()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=8192,
    messages=[{
        "role": "user",
        "content": f"""Generate a complete Python repository that reproduces the experiments in this paper.
Output all files with their full content.

Paper:
{paper_text}"""
    }]
)
```

---

## 系统对比

| 系统 | 类型 | PaperBench 分数 | 输入格式 | 开源 | 适合 batch |
|------|------|--------------|--------|------|----------|
| OpenHands | 通用 agent | **21.0%** | 文本 prompt | ✅ | ✅ headless |
| PaperCoder | 专用 paper→repo | **45.1%** | PDF→JSON | ✅ | ✅ Python API |
| DeepCode | 多 agent | 未测 | PDF / arXiv URL | ✅ | ⚠️ UI 为主 |
| Claude 直接 API | 单次生成 | 未测（模型同 OpenHands） | 文本 | ❌ | ✅ |
