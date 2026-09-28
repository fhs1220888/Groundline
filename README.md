# veritest

**可验证的试验数据分析 agent：报告里的每一个数字都能追溯到一次可复现的计算。**
*A verifiable test-data analysis agent: every number in the report traces back to a reproducible computation.*

![report](docs/report_screenshot.png)

把一次发动机热试车（或任何台架试验）的数据交给 veritest，它会完成工况分段、传感器健康检查、红线判读、阀门响应、振荡检测和仿真对比，然后写出分析报告。和一般的 "AI 分析" 不同的是：

- **数值计算只由确定性工具完成。** 大模型看不到原始数据，只负责决定调用哪些工具、怎么深挖，以及把结果写成人能读的结论。
- **每次工具调用都记入证据账本。** 账本记录参数、数据文件 SHA-256、工具源码哈希、结果和图，报告里的结论只能引用这些证据（E1、E2……）。
- **校验器逐个检查结论里的数字。** 标题和正文里的每个数字都必须能在所引用的证据中找到（允许四舍五入和 s↔ms、比例↔% 换算）。找不到的数字会在报告里用红色波浪线标出；如果用的是 LLM agent，它会收到驳回意见并有一次修正机会。
- **任何结论都可以复现。** `veritest reproduce report.json` 会用原始数据把账本里的每条证据重新执行一遍，逐条比对结果。

> 状态：v0.1 原型，数据由内置的合成器生成，还没有接入真实试车数据。

## 快速开始

```bash
pip install -e ".[all]"        # 最小安装只需要 numpy/scipy/pandas/matplotlib：pip install -e .

veritest demo                  # 生成一次带异常的合成热试车，并用规则 agent 分析
open veritest_demo/report.html
veritest reproduce veritest_demo/report.json   # 13/13 evidence entries reproduced exactly
```

分析自己的数据（CSV 的第一列为时间，也支持 NI TDMS）：

```bash
veritest analyze path/to/run.csv --reference sim.csv --limits limits.json
```

`limits.json` 的格式可以参考 `veritest synth` 生成的示例，里面包括红线、红线持续时间判据、阀门允许延迟、振荡阈值和仿真偏差容差。

## 使用大模型

规则 agent（默认）是一套固定的检查流程，不需要任何模型。换成 LLM agent 后，由模型自己规划分析步骤、交叉验证并撰写结论：

```bash
# 任何 OpenAI 兼容接口：Qwen / DeepSeek / vLLM / Ollama 本地部署都可以
export VERITEST_LLM_BASE_URL=http://localhost:11434/v1   # 例如 Ollama
export VERITEST_LLM_MODEL=qwen2.5:32b
veritest analyze run.csv --agent openai

# Anthropic
export ANTHROPIC_API_KEY=...
veritest analyze run.csv --agent anthropic --model claude-sonnet-5
```

试验数据通常很敏感，所以接口按内网私有化部署设计：模型只看到工具的输出摘要，看不到原始数据。

## 作为 MCP 服务器

```bash
veritest-mcp     # stdio 传输，配置示例见 examples/mcp_config.json
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

`veritest tools` 可以查看完整的参数说明。

## 基准测试

合成器可以注入六类已知异常，每次都附带真值：燃烧振荡、冷却剂超温、传感器冻结或缺失、测量尖峰、燃料阀延迟、室压偏低（c* 效率不足）。因此可以对 agent 做定量评测：

```bash
veritest eval --n 50 --seed 1000
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
src/veritest/
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
