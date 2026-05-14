# PaperCoder 4路模型实验统计

**总论文数**: 30 | **实验日期**: 2026-05-06 ~ 2026-05-07

> 批次日志: `experiments/runs/_batch_logs/`

## 各模型汇总

| 模型 | 成功 | .py文件 | 代码量 | 代码行数 |
|------|:----:|:------:|--------|:------:|
| DeepSeek v4-pro | 30/30 | 275 | 4.1MB | 103,985 |
| GPT-4o | 30/30 | 212 | 1.3MB | 33,481 |
| Gemini 2.5 Flash | 30/30 | 340 | 4.7MB | 100,178 |
| Claude Sonnet 4.6 | 30/30 | 464 | 11.2MB | 269,458 |

**总计**: 1291 .py 文件, 507,102 行代码

## 逐论文明细

| Paper | DeepSeek v4-pro | GPT-4o | Gemini 2.5 Flash | Claude Sonnet 4.6 |
|-------|------|------|------|------|
| adjoint-matching | 39m2s / 6f / 64KB | 28m55s / 8f / 47KB | 29m6s / 15f / 161KB | 40m8s / 12f / 289KB |
| avg-reward-pg | 24m20s / 5f / 39KB | 12m38s / 7f / 37KB | 7m34s / 7f / 52KB | 23m27s / 9f / 151KB |
| ca2-vdm | 1h59m / 15f / 245KB | 15m31s / 8f / 50KB | 23m6s / 11f / 109KB | 33m12s / 9f / 195KB |
| conformal-bayesian-quadrature | 32m6s / 7f / 63KB | 12m34s / 7f / 29KB | 14m33s / 8f / 66KB | 33m39s / 12f / 213KB |
| diffusion-convergence-rate | 30m0s / 6f / 28KB | 20m30s / 8f / 40KB | 9m39s / 6f / 50KB | 25m58s / 9f / 144KB |
| emergent-planning-rl | 1h19m / 10f / 209KB | 21m54s / 9f / 59KB | 43m51s / 18f / 318KB | - / 8f / 149KB |
| gated-attention-llm | 51m47s / 8f / 102KB | 10m40s / 6f / 30KB | 20m35s / 11f / 140KB | 49m29s / 12f / 267KB |
| generator-augmented-flows | 47m39s / 8f / 76KB | 12m4s / 6f / 32KB | 11m43s / 7f / 86KB | 39m18s / 11f / 227KB |
| hi-mar | 1h22m / 11f / 164KB | 10m35s / 6f / 40KB | 16m51s / 9f / 142KB | 52m28s / 14f / 347KB |
| lora-sb | 41m18s / 7f / 73KB | 15m8s / 6f / 34KB | 10m4s / 6f / 67KB | - / 11f / 218KB |
| luno | 1h15m / 9f / 159KB | 12m8s / 6f / 31KB | 24m22s / 9f / 169KB | 54m55s / 17f / 396KB |
| ma-rlhf | 52m8s / 7f / 156KB | 10m19s / 7f / 47KB | 25m48s / 11f / 245KB | 42m51s / 11f / 242KB |
| masked-diffusion-token-ordering | 46m37s / 8f / 121KB | 12m36s / 7f / 42KB | 20m29s / 14f / 201KB | 55m36s / 15f / 315KB |
| moe-pot | 49m50s / 7f / 89KB | 15m46s / 6f / 40KB | 20m39s / 10f / 161KB | 1h0m / 19f / 379KB |
| mrq | 1h0m / 9f / 112KB | 24m50s / 7f / 45KB | 22m21s / 9f / 117KB | 34m11s / 7f / 179KB |
| navil | 1h23m / 13f / 183KB | 18m14s / 7f / 47KB | 29m2s / 11f / 179KB | 57m51s / 17f / 377KB |
| neural-operator-flow-matching-pde | 1h5m / 10f / 117KB | 18m7s / 7f / 38KB | 36m34s / 12f / 185KB | 59m13s / 16f / 352KB |
| nfig | 1h7m / 10f / 139KB | 17m22s / 7f / 42KB | 29m34s / 10f / 137KB | 53m16s / 19f / 420KB |
| ngpt | 58m24s / 8f / 97KB | 16m38s / 6f / 36KB | 20m18s / 7f / 68KB | 32m33s / 8f / 218KB |
| olmoe | 1h23m / 14f / 206KB | 15m49s / 7f / 50KB | 23m12s / 16f / 155KB | 1h23m / 24f / 653KB |
| petl-visual-recognition | 50m5s / 8f / 102KB | 14m3s / 7f / 40KB | 23m41s / 16f / 195KB | 1h12m / 20f / 536KB |
| prioritized-generative-replay | 52m38s / 9f / 114KB | 16m57s / 10f / 61KB | 31m19s / 18f / 258KB | 1h5m / 18f / 375KB |
| pyramidal-flow-matching | 1h22m / 9f / 122KB | 15m48s / 8f / 49KB | 25m12s / 10f / 165KB | 1h8m / 20f / 544KB |
| robotic-world-model | 1h12m / 8f / 174KB | 12m11s / 6f / 40KB | 20m3s / 10f / 121KB | 1h16m / 20f / 533KB |
| sam2 | 1h55m / 16f / 285KB | 13m40s / 6f / 47KB | 47m20s / 20f / 293KB | - / 28f / 710KB |
| sc-fno | 1h23m / 12f / 177KB | 15m15s / 8f / 41KB | 36m10s / 15f / 143KB | 1h17m / 23f / 533KB |
| score | 45m17s / 7f / 93KB | 17m37s / 8f / 41KB | 23m28s / 12f / 137KB | 53m43s / 17f / 460KB |
| universal-neural-operators | 50m15s / 6f / 129KB | 13m55s / 7f / 47KB | 24m51s / 14f / 197KB | 1h16m / 20f / 519KB |
| voting-leaderboards | 49m13s / 10f / 126KB | 12m27s / 6f / 25KB | 16m4s / 8f / 93KB | 57m53s / 19f / 447KB |
| wdno | 1h31m / 12f / 208KB | 15m18s / 8f / 51KB | 32m43s / 10f / 131KB | 1h3m / 19f / 484KB |