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

> 状态：v0.1 原型。基准测试用内置合成器生成的数据；另有一次公开的真实固体发动机静态点火数据作为示例（见[真实数据示例](#真实数据示例hanaro-固体发动机静态点火)），还没有在液体发动机真实试车数据上验证过。

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
| `pulse_metrics` | 脉冲型通道（如固体发动机推力）的峰值、工作时间、积分（总冲）和平均值 |
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

总体精确率为 100%，4 次正常试车上的误报为 0，134/134 条结论通过校验，942/942 个数字可溯源。

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

### 多模型排行榜

`groundline leaderboard` 让多个 agent / 模型跑同一组合成试车，并汇总成一张表（`leaderboard/LEADERBOARD.md`）。模型列表写在 JSON 里，示例见 `examples/leaderboard.json`：规则 agent、gpt-5.6-sol，以及通过 [Ollama](https://ollama.com) 在本地运行的 Qwen2.5 7B / 14B。已有的 `groundline eval` 结果可以用 `"from"` 直接导入，不必重跑。结果按模型分别保存，中断后再次运行会接着跑没跑完的模型。

```bash
groundline leaderboard examples/leaderboard.json            # 全部模型
groundline leaderboard examples/leaderboard.json --only "qwen2.5-7b (本地)"
```

表里最关键的一列是**初稿中无出处的数字**：校验器第一次拦下、在所引用证据里找不到的数字，也就是没有校验器时会直接进入报告的数字。检出率按全部试车计算，模型崩溃或没交报告的试车里的异常都算漏检。

2026-09-28 的结果（`--n 14 --seed 1000`，本地模型在 16 GB 内存的 MacBook Pro 上用 Ollama 运行，上下文 16k；原始结果在 `docs/results/leaderboard/`）：

| 模型 | 交出报告 | 检出率（严格 / 宽松） | 精确率 | 正常试车误报 | 初稿中无出处的数字 | 校验后仍无出处 | 秒/次 |
|---|---|---|---|---|---|---|---|
| 规则 agent | 14/14 | 100% / 100% | 100% | 0 | – | 0 | 1.3 |
| gpt-5.6-sol | 14/14 | 100% / 100% | 95% | 0 | 0 / 376（0%） | 0 | 22 |
| Qwen2.5 7B | 7/14 | 0% / 15% | 0% | 0 | 10 / 80（12%） | 8 个数字、4 条结论 | 346 |
| Qwen2.5 3B | 13/14 | 0% / 0% | – | 0 | 1 / 1 | 1 个数字、4 条结论 | 28 |

我们逐条核对了 Qwen2.5 7B 初稿里被拦下的 10 个数字（用同样的种子重算证据）：

- **4 个是编造的数值**：把 49.9994 kg 的流量积分写成"近 60"；把实际约 10.3% 的最大偏差写成 2.29%；两处凭空写出的时长（"0.86 秒""持续 2 秒"）。
- **6 个是真实数值、但引用了错误的证据**：其中 2 个（主级段起止时刻）被拿来支撑一个错误结论，说冷却剂温度在整个主级段都超限，而实际越限只有 3.95–5.91 s。
- **没有误报**：被拦下的每个数字都对应一个真实的问题。3B 的结果里有 1 处误报，把试车编号"SYN-1013"当成了数值，已修正。

这组结果也说明了校验器的边界：

- **修正轮救不回小模型。** 7B 用了 5 次修正轮，校验后仍有 8 个数字没有出处。校验器的作用是把问题标出来（报告里的红色波浪线），不能指望它让模型改对。
- **它只检查"数字有没有出处"，不检查"用得对不对"。** 7B 把稳态的氧化剂流量说成"尖脉冲"，引用的数字全部真实，校验器无法发现。另有一处"总计 11 秒"是编的，却碰巧和证据里的某个数在 1% 以内对上，被放行了。
- **小模型的主要问题是做不完任务，而不只是编数字。** 7B 有 5 次在 24 步内没交出报告，2 次请求超时；3B 大多交了空报告，不说话自然也就不编造。两次重跑之间，7B 交出报告的次数从 11 次变成 7 次，随机性很大，样本也小，这些数字只能作为粗略参考。

**这组数字要打折扣看。** 规则 agent 是和这个合成器一起调出来的，满分只能说明管线自洽，不能说明它能处理真实数据。这个基准真正的用途是：

1. 比较不同 LLM agent 的表现，以及它们编造数字的频率；
2. 修改工具时防止回归；
3. 接入更难的场景，比如多异常耦合、真实噪声谱、传感器漂移，以及和真实试车数据的对照。

## 真实数据示例：HANARO 固体发动机静态点火

`examples/hanaro_knsb/` 里是首尔大学火箭队 HANARO 公开的一次 KNSB 固体发动机静态点火（2025 年，数据来自 [snu-hanaro/static-fire-toolkit](https://github.com/snu-hanaro/static-fire-toolkit)，MIT 许可）。原始文件原样放在 `raw/`，`prepare.py` 把它们整理成 Groundline 的输入：

- 推力来自称重传感器，采集时间不均匀、有重复时间戳。去重后插值到 100 Hz 网格，离最近原始样本超过 25 ms 的点保留为 NaN，不掩盖丢帧；
- 公开资料里没有称重传感器的标定常数，所以电压到牛顿的换算用直线拟合 HANARO 自己处理后的推力曲线（R² = 0.9999）；
- 压力来自一台独立的约 10 Hz 记录仪，时钟和推力采集卡不同步。时钟偏差取压力与推力相关性最大的平移量（+5.652 s，相关系数 0.9995）；
- HANARO 没有公布壳体压力或推力的限值，所以不设红线，也没有仿真预测。

```bash
python examples/hanaro_knsb/prepare.py        # 重新生成 run.csv / meta.json / limits.json
groundline analyze examples/hanaro_knsb/run.csv
```

规则 agent 的结果和 HANARO 自己的处理结果对照：

| | Groundline | HANARO 处理结果 |
|---|---|---|
| 推力峰值 | 2222.2 N | 2221.7 N |
| 总冲 | 6362 N·s（按峰值 10% 截取，3.91 s）<br>6391 N·s（按 2% 截取，4.12 s） | 6411 N·s（其截取窗口 4.35 s） |

另外发现了两处测量问题，都不在点火段内：推力通道有 285 段数据缺失，间隔中位数 0.67 s，规律性很强，更像采集系统周期性丢帧；点火前 121 s 左右推力通道有一个孤立尖峰。

LLM agent（gpt-5.6-sol）在这组数据上给出的 3 条结论与规则 agent 一致（时序、推力性能、推力通道的丢帧和尖峰），25 个数字全部可溯源，初稿没有编造数字。它还主动说明了 Pc 只有 10 Hz、不能用来判断振荡，改用推力通道检查 5–50 Hz 频段，没有发现振荡。4 次请求，约 1.3 万输入 token。

这组数据暴露了几处只按液体发动机合成数据设计的假设，已经修正：
- 慢速记录的通道被插值到快网格上后，振荡检测会去找超过其奈奎斯特频率的成分，尖峰检测会把每个真实采样点都当成尖峰。现在工具会读取 `meta.json` 里的 `native_rate_hz`，在原始采样率上判断；
- 量化后静止的信号（如点火前稳定的环境压力）残差中位数为 0，噪声底被算成 0，一格的跳动就被当成尖峰；
- 点火前后传感器读数本来就不变，不应报"信号冻结"。现在只报告与点火段重叠的冻结；
- 反复出现的丢帧会被合并成一条结论，而不是几百条；
- 点火时推力的快速爬升会泄漏到振荡检测频段的最低一格，被误报成"5 Hz 振荡"。现在峰值落在频段最低一格的窗口不计入；
- 新增 `pulse_metrics` 工具计算推力峰值、工作时间和总冲。

改动后，合成基准测试上规则 agent 的检出率、精确率仍是 100%，0 误报。

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
