# Groundline

**可验证的试验数据分析 agent：报告里的每一个数字都能追溯到一次可复现的计算。**
*Verifiable AI agent for engine test data: every number in the report traces back to a reproducible computation.*

名字取自 *grounded*（每个数字都有依据）和 *redline*（红线判读）。

![report](docs/report_screenshot.png)

把一次发动机热试车（或任何台架试验）的数据交给 Groundline，它会完成工况分段、传感器健康检查、红线判读、阀门响应、振荡检测和仿真对比，然后写出分析报告。和一般的 "AI 分析" 不同的是：

- **数值计算只由确定性工具完成。** 大模型看不到原始数据，只负责决定调用哪些工具、怎么深挖，以及把结果写成人能读的结论。
- **每次工具调用都记入证据账本。** 账本记录参数、数据文件 SHA-256、工具源码哈希、结果和图，报告里的结论只能引用这些证据（E1、E2……）。
- **校验器逐个检查结论里的数字。** 标题和正文里的每个数字都必须能在所引用的证据中找到（允许四舍五入和 s↔ms、比例↔% 换算）。找不到的数字会在报告里用红色波浪线标出；如果用的是 LLM agent，它会收到驳回意见并有一次修正机会。
- **任何结论都可以复现。** `groundline reproduce report.json` 会用原始数据把账本里的每条证据重新执行一遍，逐条比对结果。

> 状态：v0.1 原型，数据由内置的合成器生成，还没有接入真实试车数据。

## 快速开始

```bash
pip install -e ".[all]"        # 最小安装只需要 numpy/scipy/pandas/matplotlib：pip install -e .

groundline demo                  # 生成一次带异常的合成热试车，并用规则 agent 分析
open groundline_demo/report.html
groundline reproduce groundline_demo/report.json   # 13/13 evidence entries reproduced exactly
```

分析自己的数据（CSV 的第一列为时间，也支持 NI TDMS）：

```bash
groundline analyze path/to/run.csv --reference sim.csv --limits limits.json
```

`limits.json` 的格式可以参考 `groundline synth` 生成的示例，里面包括红线、红线持续时间判据、阀门允许延迟、振荡阈值和仿真偏差容差。

## 使用大模型

规则 agent（默认）是一套固定的检查流程，不需要任何模型。换成 LLM agent 后，由模型自己规划分析步骤、交叉验证并撰写结论。

最省事的方式是在项目根目录放一个 `.env`（已在 `.gitignore` 中，不会被提交）：

```bash
cp .env.example .env      # 填入 OPENAI_API_KEY，按需改 GROUNDLINE_AGENT / GROUNDLINE_LLM_MODEL
groundline config         # 查看当前生效的配置（key 会打码）
groundline demo           # 之后所有命令默认使用 .env 里的 agent 和模型
```

Groundline 会从当前目录向上查找 `.env`；命令行参数和 shell 里已设置的环境变量优先于 `.env`。也可以不用 `.env`，直接用环境变量或参数：

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
groundline analyze run.csv --agent openai --model gpt-4o-mini

# 任何 OpenAI 兼容接口：Qwen / DeepSeek / vLLM / Ollama 本地部署都可以
export GROUNDLINE_LLM_BASE_URL=http://localhost:11434/v1   # 例如 Ollama
export GROUNDLINE_LLM_MODEL=qwen2.5:32b
groundline analyze run.csv --agent openai

# Anthropic
export ANTHROPIC_API_KEY=...
groundline analyze run.csv --agent anthropic --model claude-sonnet-5
```

评测 LLM agent 时，除了检出率，还会统计**初稿中被校验器拦下的编造数字**，以及修正后的结果和 token 用量：

```bash
groundline eval --agent openai --model gpt-4o-mini --n 10 --lang en --out eval_openai.json
```

连接官方 OpenAI 时自动使用 Responses API（推理模型如 `gpt-5.6-sol` 只有在这个接口上才能同时推理和调用工具），推理强度用 `GROUNDLINE_LLM_REASONING_EFFORT` 设置；连接其他 OpenAI 兼容服务时使用 Chat Completions，可用 `GROUNDLINE_OPENAI_API=responses|chat` 强制指定。默认不传 temperature，需要时用 `GROUNDLINE_LLM_TEMPERATURE` 设置。

试验数据通常很敏感，所以接口按内网私有化部署设计：模型只看到工具的输出摘要，看不到原始数据。

## 作为 MCP 服务器

```bash
groundline-mcp     # stdio 传输，配置示例见 examples/mcp_config.json
```

它提供 `open_run`、`list_analysis_tools`、`run_analysis`、`verify`、`write_html_report` 五个工具。任何 MCP 客户端（Claude Desktop、Cursor 或你自己的 agent）都可以充当规划者，校验规则保持不变。

## 分析工具

| 工具 | 作用 |
|---|---|
| `describe_data` | 通道、采样率、单位、NaN 统计 |
| `segment_phases` | 基于室压 10%/90% 划分预试、启动、主级、关机、后处理五个阶段，并记录阀门指令时刻 |
| `check_sensor_health` | NaN 缺失、信号冻结、孤立尖峰（局部稳健 σ） |
| `check_redlines` | 红线判读，带持续时间判据和 5 ms 平滑，短毛刺记为被抑制的瞬态 |
| `measure_valve_response` | 从阀门指令到响应通道起跳的延迟，与允许值比较 |
| `detect_oscillation` | 滑动窗口 FFT，给出窄带振荡的频率、幅值（% of mean）和起止时间 |
| `compare_reference` | 与仿真预测对比：平均偏差、RMSE，以及持续超差的时间段 |
| `channel_stats` / `plot_window` | 深挖用的统计和作图 |

`groundline tools` 可以查看完整的参数说明。

## 基准测试

合成器可以注入六类已知异常，每次都附带真值：燃烧振荡、冷却剂超温、传感器冻结或缺失、测量尖峰、燃料阀延迟、室压偏低（c* 效率不足）。因此可以对 agent 做定量评测：

```bash
groundline eval --n 50 --seed 1000
```

规则 agent 在 50 次随机试车上的结果如下：

| 异常 | n | 检出率 | 分类正确率 | 起始时刻误差（中位数） |
|---|---|---|---|---|
| oscillation | 15 | 100% | 100% | 12 ms |
| overtemp | 22 | 100% | 100% | 0 ms |
| sensor_dropout | 10 | 100% | 100% | 0 ms |
| sensor_spike | 8 | 100% | 100% | 0 ms |
| valve_delay | 15 | 100% | 100% | 0 ms |
| pc_deficit | 14 | 100% | 100% | 112 ms |

总体精确率为 100%，4 次正常试车上的误报为 0，135/135 条结论通过校验，947/947 个数字可溯源。

LLM agent（`gpt-5.6-sol`，Responses API）在同一组种子的前 14 次试车上（`--n 14 --seed 1000`，含 2 次正常试车，6 类异常全覆盖）：

| 指标 | 规则 agent | gpt-5.6-sol |
|---|---|---|
| 检出率 / 分类正确率 | 100% / 100% | 100% / 100% |
| 精确率 | 100% | 95%（1 条误报） |
| 正常试车上的误报 | 0 | 0 |
| 初稿中编造的数字 | — | 0 / 376 |
| 每次试车 | 1.6 s | 22 s，约 1.9 万输入 / 1,400 输出 token |

唯一的误报是把阀门延迟引起的冷却剂温升滞后单独报成了一条性能偏差，没有归到阀门延迟下面。样本只有 14 次、只跑了一轮，模型输出本身有随机性，这组数字只能作为粗略参考。

第一次评测时精确率只有 36%：37 条"误报"里有 35 条其实是"阀门响应合格""未检出振荡"这类检查通过的结论，被模型填成了异常类别。提示词里补上"非 observation 类别表示发现了异常，检查通过的结论用 observation"之后，精确率升到 95%。评测标准没有改动。

**这组数字要打折扣看。** 规则 agent 是和这个合成器一起调出来的，满分只能说明管线自洽，不能说明它能处理真实数据。这个基准真正的用途是：

1. 比较不同 LLM agent 的表现，以及它们编造数字的频率；
2. 修改工具时防止回归；
3. 接入更难的场景，比如多异常耦合、真实噪声谱、传感器漂移，以及和真实试车数据的对照。

## 设计要点

- **LLM 不做算术。** 系统提示明确要求结论里的每个数字都来自证据。比如模型想说"偏差 5%"，就必须调用 `compare_reference` 拿到这个数，而不是自己用两个数相减。
- **规则 agent 也要过校验。** 开发过程中，校验器抓到了规则模板里写死的"5 个采样点"：这个判据当时没有出现在证据里。修复方法是把判据写进工具输出，而不是放宽校验。
- **会归因的结论。** 如果流量和预测一致而室压偏低，报告会指向 c* 效率不足；如果燃料阀开得晚导致点火推迟，由此产生的温升滞后会被归到阀门延迟下面，而不是单独报成一个问题。

## 目录结构

```
src/groundline/
  synth.py        合成热试车数据和异常真值
  io.py           CSV / TDMS 读写
  session.py      Session 与证据账本
  tools.py        确定性分析工具（全部数值来源）
  findings.py     结论数据结构与数字溯源校验器
  agent.py        规则 agent、LLM agent（OpenAI 兼容 / Anthropic）
  report.py       自包含 HTML 报告和 report.json
  reproduce.py    证据复现
  evaluate.py     基准评测
  mcp_server.py   MCP 服务器
  config.py       .env 配置加载
  cli.py          命令行
```

## 路线图

- [ ] 在真实试车数据（脱敏后）上验证判据，校准阈值
- [ ] 偏差归因：仿真和实测对不上时，按机理、网格、边界条件、制造偏差、传感器分别列出候选原因并设计验证
- [ ] 多次试车横向对比与趋势分析
- [ ] 更难的合成场景和 LLM agent 排行榜
- [ ] 试验报告模板（Word/PDF 导出）

## License

MIT
