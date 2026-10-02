# Groundline

English | [简体中文](https://github.com/fhs1220888/Groundline/blob/main/README.zh-CN.md)

<!-- mcp-name: io.github.fhs1220888/groundline -->

**A verifiable AI agent for engine test data: every number in the report traces back to a reproducible computation.**

**[Live demo](https://fhs1220888.github.io/Groundline/demo/)**: try to get a wrong number past the verifier, in your browser, on real static-fire evidence.

The name comes from *grounded* (every number has a source) and *redline* (limit checks).

![report](https://raw.githubusercontent.com/fhs1220888/Groundline/main/docs/report_screenshot.png)

Give Groundline the data from an engine hot-fire (or any bench test) and it segments the run into phases, checks sensor health, redlines, valve response and oscillations, compares against the simulation prediction, and writes an analysis report. Unlike a typical "AI analysis":

- **All numbers come from deterministic tools.** The LLM never sees raw samples. It only decides which tools to call and how to dig deeper, and turns their results into readable findings.
- **Every tool call goes into an evidence ledger.** The ledger records parameters, the data file's SHA-256, a hash of the tool's source code, the result and the figure. Findings may only cite these entries (E1, E2, ...).
- **A verifier checks every number in every finding.** Each number in a title or statement must be found in the evidence it cites (rounding and s↔ms, fraction↔% conversions allowed). Numbers that are not found get a red squiggle in the report, and an LLM agent gets the rejection back with one chance to fix it. The verifier also checks what a number *means*: the unit written after it (s, Hz, %, bar, N·s, ...) must match the unit of the evidence field, and a number introduced by a word such as "peak", "duration", "mean" or "impulse" must come from a field with that role (see [Semantic verification](#semantic-verification)).
- **Every finding can be reproduced.** `groundline reproduce report.json` re-runs every ledger entry on the raw data and compares the results one by one.

> Status: v0.1 prototype. Benchmarks use data from the built-in synthetic generator. Public real tests are worked examples: a solid-motor static fire ([HANARO](#real-data-example-hanaro-solid-motor-static-fire)), a liquid-engine hot fire ([Triton](#real-data-example-triton-liquid-engine-hot-fire)) and four hybrid-motor hot fires checked against the team's own reports ([UVic MULE-1](#real-data-example-uvic-mule-1-hybrid-motor-checked-against-the-teams-own-reports)). None comes with redlines, valve commands or a simulation prediction, so those checks are still validated on synthetic data only.

## Quick start

```bash
pip install -e ".[all]"        # minimal install needs only numpy/scipy/pandas/matplotlib: pip install -e .

groundline demo                  # generate a synthetic hot-fire with anomalies and analyse it with the rule agent
open groundline_demo/report.html
groundline reproduce groundline_demo/report.json   # 13/13 evidence entries reproduced exactly
```

Analyse your own data (CSV with time in the first column; NI TDMS is supported too):

```bash
groundline analyze path/to/run.csv --reference sim.csv --limits limits.json
```

For the `limits.json` format, see the example written by `groundline synth`: redlines, redline persistence, allowed valve latency, oscillation thresholds and the tolerance for deviation from the simulation.

A raw DAQ export rarely has Groundline's layout. `groundline ingest` turns it into one from a small JSON mapping, and drafts the mapping for you:

```bash
groundline ingest raw_log.csv                         # writes raw_log/map.json and lists what it guessed
groundline ingest raw_log.csv --map raw_log/map.json  # writes raw_log/run.csv + meta.json
groundline analyze raw_log/run.csv
```

The draft reads units from headers such as `Chamber Pressure (psi)` or from a units row above the names (it also realigns a units row that is a cell off), finds the time column, names the chamber pressure `Pc`, sums several thrust load cells into `F_thrust`, and assigns each channel a kind from its unit and name. In the mapping you then set what it cannot know: names, a time window, a uniform grid (`grid_hz`) and how long a gap may be bridged (`max_gap_s`, beyond which the run keeps NaN so dropouts stay visible), plus `scale` / `offset` per channel. The Triton and UVic examples below are converted this way, and reproduce the earlier hand-written conversions exactly. Read the draft before trusting it, and keep test outcomes out of `description`: LLM agents read it.

## Using an LLM

The rule agent (the default) is a fixed checklist and needs no model. With an LLM agent, the model plans the analysis, cross-checks and writes the findings itself.

The easiest setup is a `.env` in the project root (it is in `.gitignore` and never committed):

```bash
cp .env.example .env      # fill in OPENAI_API_KEY; change GROUNDLINE_AGENT / GROUNDLINE_LLM_MODEL as needed
groundline config         # show the configuration in effect (keys masked)
groundline demo           # from now on every command uses the agent and model from .env
```

Groundline looks for `.env` from the current directory upwards; command-line arguments and variables already set in the shell take precedence. You can also skip `.env` and use environment variables or arguments:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
groundline analyze run.csv --agent openai --model gpt-4o-mini

# Any OpenAI-compatible endpoint: Qwen, DeepSeek, vLLM, a local Ollama ...
export GROUNDLINE_LLM_BASE_URL=http://localhost:11434/v1   # e.g. Ollama
export GROUNDLINE_LLM_MODEL=qwen2.5:32b
groundline analyze run.csv --agent openai

# Anthropic
export ANTHROPIC_API_KEY=...
groundline analyze run.csv --agent anthropic --model claude-sonnet-5-5
```

When benchmarking an LLM agent, besides detection rates Groundline also counts **the invented numbers the verifier caught in the first draft**, the result after the fix round, and token usage:

```bash
groundline eval --agent openai --model gpt-4o-mini --n 10 --lang en --out eval_openai.json
```

With the official OpenAI endpoint the Responses API is used automatically (reasoning models such as `gpt-5.6-sol` can only reason and call tools together there); set the reasoning effort with `GROUNDLINE_LLM_REASONING_EFFORT`. Other OpenAI-compatible services use Chat Completions; force either with `GROUNDLINE_OPENAI_API=responses|chat`. No temperature is sent by default; set one with `GROUNDLINE_LLM_TEMPERATURE`. With `--agent anthropic` the same variable sets the effort level (`low` … `max`); the prompt prefix is cached between agent steps.

Test data is usually sensitive, so the interface is built for on-premise deployment: the model only sees summaries of tool outputs, never the raw data.

## As an MCP server

Groundline is listed in the [MCP Registry](https://registry.modelcontextprotocol.io) as `io.github.fhs1220888/groundline`. Clients that install from the registry run it with `uvx groundline mcp`; to add it by hand (Claude Desktop, Cursor, ...):

```json
{
  "mcpServers": {
    "groundline": { "command": "uvx", "args": ["groundline", "mcp"] }
  }
}
```

From a local checkout, `groundline mcp` (or `groundline-mcp`) runs the same server on stdio.

It exposes five tools: `open_run`, `list_analysis_tools`, `run_analysis`, `verify` and `write_html_report`. Any MCP client (Claude Desktop, Cursor or your own agent) can act as the planner; the verification rules stay the same.

## Analysis tools

| Tool | What it does |
|---|---|
| `describe_data` | Channels, sample rate, units, NaN counts |
| `segment_phases` | Splits the run into pre-test, startup, mainstage, shutdown and post-test at 10 % / 90 % of chamber pressure, and records valve command times |
| `check_sensor_health` | NaN gaps, frozen signals, isolated spikes (local robust σ) |
| `check_redlines` | Redline checks with a persistence criterion and 5 ms smoothing; short glitches are counted as suppressed transients |
| `measure_valve_response` | Latency from a valve command to the response channel moving, compared with the allowed value |
| `detect_oscillation` | Sliding-window FFT: frequency, amplitude (% of mean) and start/end of narrow-band oscillations |
| `compare_reference` | Comparison with the simulation prediction: mean deviation, RMSE and intervals of sustained deviation |
| `pulse_metrics` | Peak, action time, integral (total impulse) and mean of a pulse-shaped channel such as solid-motor thrust |
| `check_thrust_pressure_ratio` | Thrust over chamber pressure (both above baseline) is proportional to thrust coefficient × throat area; flags a change of more than 50 % while the engine burns (throat erosion or failure, or a failing sensor) |
| `channel_stats` / `plot_window` | Statistics and plots for drilling down |

`groundline tools` prints the full parameter descriptions.

## Benchmarks

The synthetic generator can inject six kinds of known anomalies, always with ground truth: combustion oscillation, coolant over-temperature, a frozen or missing sensor, measurement spikes, a late fuel valve, and low chamber pressure (low c* efficiency). That makes agents measurable:

```bash
groundline eval --n 50 --seed 1000
```

Rule agent on 50 random runs:

| Anomaly | n | Recall | Category accuracy | Median start-time error |
|---|---|---|---|---|
| oscillation | 15 | 100% | 100% | 12 ms |
| overtemp | 22 | 100% | 100% | 0 ms |
| sensor_dropout | 10 | 100% | 100% | 0 ms |
| sensor_spike | 8 | 100% | 100% | 0 ms |
| valve_delay | 15 | 100% | 100% | 0 ms |
| pc_deficit | 14 | 100% | 100% | 112 ms |

Overall precision is 100%, with 0 false positives on the 4 nominal runs; 134/134 findings verified and 943/943 numbers grounded.

These six anomaly types and the detectors were written together, so the table above mostly shows that the tools catch what the generator was built to inject. Two further benchmarks, after the LLM results below, take that away step by step.

The LLM agent (`gpt-5.6-sol`, Responses API) on the first 14 runs of the same seeds (`--n 14 --seed 1000`, including 2 nominal runs and all six anomaly types):

| Metric | Rule agent | gpt-5.6-sol |
|---|---|---|
| Recall / category accuracy | 100% / 100% | 100% / 100% |
| Precision | 100% | 95% (1 false positive) |
| False positives on nominal runs | 0 | 0 |
| Invented numbers in first drafts | — | 0 / 376 |
| Per run | 1.6 s | 22 s, about 19k input / 1,400 output tokens |

The one false positive reports the coolant temperature lag caused by the late valve as a separate performance deviation instead of attributing it to the valve delay. With only 14 runs, one pass, and non-deterministic model output, treat these numbers as a rough guide.

The first evaluation scored only 36% precision: 35 of the 37 "false positives" were passed checks such as "valve response within limits" or "no oscillation detected" that the model had filed under an anomaly category. After adding one rule to the prompt ("a category other than observation means an anomaly was found; checks that passed use observation"), precision rose to 95%. The scoring was not changed.

### Source tags with a real model

The same 14 synthetic runs (seeds 1000–1013, English, gpt-5.6-sol at low reasoning effort), once with the instruction to tag every number with its source field and once without:

| | Tags on | Tags off |
|---|---|---|
| Recall / precision | 100% / 91% (2 false positives) | 100% / 95% (1 false positive) |
| False positives on nominal runs | 0 | 0 |
| Numbers tagged in final reports | 236 / 240 (98%) | — |
| Wrong tags in first drafts | 0 | — |
| Numbers stopped in first drafts | 0 / 242 | 0 / 314 |
| Tokens, 14 runs | 277k input / 18.5k output | 296k input / 17.5k output |
| Time per run | 21 s | 20 s |

Tagging cost nothing measurable, and the model tagged almost every number correctly, so each of its numbers is checked against exactly one field (in the planted-error benchmark below, misplaced values are caught 67% of the time untagged and 99.9% tagged). The three false positives are the same known mistake, a coolant-temperature lag caused by the late valve reported as a deviation of its own: one run (seed 1010) in both arms, and one more run (seed 1001) only with tags. With 14 runs that is not a measurable difference. Tagged reports carry fewer numbers (240 against 312).

The test caught two mistakes on our side before they reached these numbers. The model wrote tags such as `E3.result.issues[0].count`, because its tool results arrive wrapped in `result`, and 6 of 11 tags in a smoke run were rejected; such paths are accepted now, and a wrong tag's message names the field the value does match. And "1.127 kHz" for a 1126.95 Hz oscillation was flagged, because kHz was not a known unit; it is now.

### The LLM agent on the real tests

One gpt-5.6-sol run with tags on each real test. Compared with the team reports and the rule agent:

- UVic 2024-12-12: the record ends while chamber pressure is still rising, so performance and stability cannot be assessed; a coincident multi-channel transient at the cut-off; a dead thermocouple. As the team reported.
- UVic 2025-01-18: the chamber-pressure offset, the dead chamber thermocouples, the duplicated pressure channels, and F/Pc stable through mainstage but diverging after it.
- UVic 2025-02-08: it flags the failed F/Pc consistency check but calls it "sensor-confounded" (the chamber pressure has a large offset) rather than pointing to the throat. The rule agent names a throat change as one possible cause; neither can say what broke.
- UVic 2025-09-20: the saturated chamber pressure, three dead thermocouples, and transients near shutdown that make the peak not credible.
- Triton: the shared DAQ dropouts, the saturated regulator sensor, the chamber pressure that does not recover after shutdown, the isolated spikes, and the 127 Hz event as a warning.
- HANARO: a nominal firing, off-fire gaps on the thrust channel.

All findings verify. Two corrections went into getting here:

- **A leak in our own test.** The first UVic runs used metadata whose description quoted each team report ("nozzle throat insert broke ..."), and `describe_data` hands the description to the model; its 2025-02-08 report then said "consistent with the reported nozzle/throat hardware failure". The reports are now kept out of the metadata (they stay in `prepare.py` and here), and the results above are from clean reruns. The rule agent never reads the description, so its results were not affected.
- **Two prompt rules**, after the first Triton run left out the chamber pressure that does not recover after shutdown (although the sensor-health result it read reported it) and rated a 0.58% oscillation critical: every sensor-health issue must reach the report, and an oscillation is critical only at twice its criterion. On the 14 synthetic runs the new prompt keeps recall at 100% and precision at 95% (1 false positive, the known coolant-lag mistake), with 97% of numbers tagged and no wrong tags.

Earlier, two runs needed a fix round for the same verifier false alarm: "its maximum 5180.25 psi" tagged to a `stuck_value` field. A tagged number now also matches sibling fields of the same object that hold the very same value, and stuck-at-maximum issues carry a `channel_max` field.

### Realistic suite: faults and nuisances from real logs

`groundline eval --suite realistic` adds seven faults found in the Triton and UVic logs, each with ground truth (DAQ dropouts on every channel, a saturated transducer, chamber pressure stuck high after shutdown, a zero offset below vacuum, a duplicated channel, a dead channel, 60 Hz mains hum), and nuisances that must not be reported (a start-up overshoot below the redline, water hammer at shutdown, ADC quantization). The classic suite is unchanged seed for seed. Rule agent, 100 runs (`--n 100 --seed 2000`): 100% recall, precision and category accuracy over all 13 types, 0 false positives on the 15 nominal runs, 257/257 findings verified ([details](docs/results/realistic_suite_rule.json)).

Building the suite exposed three gaps that are now fixed: mains hum was reported as combustion instability (a line already present before ignition is now reported as pickup), chamber pressure stuck after shutdown went unnoticed when the record ends soon after shutdown, and a duplicated channel also produced a false performance deviation. Like the first table, though, these faults were written after the fixes, so the suite is a regression test more than a measurement.

### Hybrid benchmark: known anomalies in real logs

`groundline hybrid-bench` keeps the background real and makes only the anomaly synthetic: it injects one anomaly of known size and time into a real firing, runs the rule agent, and compares with the same configuration on the untouched log, so the log's own real problems are neither credited nor counted against it (an injection that lands on one of them is "masked"). Backgrounds: the Triton hot fire and UVic MULE-1 2025-01-18 (HANARO cannot serve: its chamber pressure was logged at 10 Hz and its thrust channel has dropouts across the whole record). 5 injections per size and background, 270 cases ([details](docs/results/hybrid_bench.json)):

| Injected | Detected by size | |
|---|---|---|
| Oscillation on Pc (zero-to-peak, % of steady Pc; criterion 0.5%) | Triton: 0.25% 0/5, 0.5% 1/4, 1% 5/5, 2% 4/4, 4% 5/5 · UVic: 0.25–1% 0/15, 2% 2/5, 4% 4/5 | |
| Pc above a redline (time above; persistence 10 ms) | 5 ms 0/10, 10 ms 0/10, 20 ms 10/10, 50 ms 10/10, 200 ms 10/10 | |
| Pc below a prediction (% of steady Pc; tolerance 2%) | Triton: 1% 0/5, 2% 0/5, 3–8% 15/15 · UVic: masked (its Pc carries a known zero offset) | |
| One-sample spike on thrust (local noise sigmas; criterion 12) | 6σ 0/10, 12σ 3/10, 24σ 9/10, 48σ 8/10 | |
| Gap in the thrust channel | 10 ms to 1 s: 40/40 | |
| Thrust channel frozen (minimum 50 ms) | 20 ms 4/10, 50 ms to 1 s: 30/30 | |

**No injection produced a new false positive in 270 cases.** The redline, deviation, gap and freeze checks behave as specified on real noise. Oscillation detection depends on the engine: on the Triton liquid engine it works from about the 0.5% criterion upwards; on the rough-burning UVic hybrid, broadband combustion noise hides lines below a few percent. A second search pass with 0.5 s windows (added for this) raised UVic from 2/10 to 6/10 at 2–4%. The misses at 24σ and 48σ are spikes placed in Triton's start-up pressure event or next to a DAQ dropout, where spike detection is deliberately desensitised; the four 20 ms "detections" are the spike check firing at the step back, not the flatline check.

### Model leaderboard

`groundline leaderboard` runs several agents / models on the same synthetic runs and summarises them in one table (`leaderboard/LEADERBOARD.md`). Models are listed in a JSON file; `examples/leaderboard.json` has the rule agent, gpt-5.6-sol and Qwen2.5 7B / 3B running locally through [Ollama](https://ollama.com) (14B needs more memory). An existing `groundline eval` result can be imported with `"from"` instead of re-running it. Results are saved after every run, so an interrupted leaderboard resumes where it stopped.

```bash
groundline leaderboard examples/leaderboard.json            # all models
groundline leaderboard examples/leaderboard.json --only "qwen2.5-7b (本地)"
```

The key column is **numbers without a source in the first draft**: numbers the verifier stopped on the first submission because they are not in the cited evidence, i.e. numbers that would have gone into the report without a verifier. Recall is computed over all runs: anomalies in runs where the model crashed or submitted nothing count as missed.

Results from 2026-09-29 (`--n 14 --seed 1000`; local models ran with Ollama on a 16 GB MacBook Pro with a 16k context; raw results in `docs/results/leaderboard/`):

| Model | Reports submitted | Recall (strict / loose) | Precision | FPs on nominal runs | First-draft numbers without a source | Still wrong after verification | s/run |
|---|---|---|---|---|---|---|---|
| Rule agent | 14/14 | 100% / 100% | 100% | 0 | – | 0 | 1.3 |
| gpt-5.6-sol | 14/14 | 100% / 100% | 95% | 0 | 0 / 376 (0%) | 0 | 22 |
| Qwen2.5 7B | 11/14 | 5% / 50% | 5% | 0 | 25 / 98 (26%), plus 2 with the wrong meaning | 26 numbers, 19 findings | 478 |
| Qwen2.5 3B | 13/14 | 0% / 0% | – | 0 | 1 / 1 | 1 number, 4 findings | 28 |

Strict recall needs the finding's channel field and time window to match; loose recall also counts a finding with no channel if its category is right and its time does not conflict (the model found the problem but did not fill in the structured field).

Qwen2.5 7B was run three times with very different results (reports submitted 11 / 7 / 11; first-draft numbers without a source 15% / 12% / 23%, as scored at the time). The table shows the last run, re-scored with the current verifier (26%); it is the first one that stored its evidence ledger and could be re-verified with the new verifier.

On the second run we audited, one by one, the 10 numbers the verifier stopped in the 7B first drafts, recomputing the evidence from the same seeds:

- **4 were invented values**: a flow integral of 49.9994 kg written as "about 60"; a maximum deviation of about 10.3% written as 2.29%; two durations with no source ("0.86 s", "lasting 2 s").
- **6 were real values cited from the wrong evidence**: 2 of them (the mainstage start and end times) propped up a wrong claim that the coolant was over its redline for the whole mainstage, when it was only over from 3.95 to 5.91 s.
- **No false alarms**: every stopped number was a real problem. The 3B run had one false alarm, the test ID "SYN-1013" read as a number; fixed.

On the last run, semantic verification stopped 4 more numbers that *have* a source but the wrong meaning; the grounding-only verifier would have passed all of them. All 4 are real problems:

- "lasting about 0.2 s": 0.2 is the value of a flow integral, not a duration. This is exactly the "made-up number that happens to match an unrelated one" loophole.
- "mean deviation 0.069%": the 0.069 in the evidence is an absolute deviation in MPa, not a percentage.
- Two flow integrals given in "kg·s": the integral of a flow is a mass in kg.

The first time it ran on real output, semantic verification flagged 4 more numbers; the audit showed they were bugs in the verifier itself: it read compound units such as "MPa·s" and "kg/s·s" only up to the first unit. Compound units are now read whole and cancelled (kg/s·s = kg). After the fix there are still no false alarms on the 60 numbers written by gpt-5.6-sol or the 1,884 written by the rule agent.

These results also show the verifier's limits:

- **The fix round does not rescue small models.** On the last run 7B used 9 fix rounds and still had 26 problem numbers after verification. The verifier's job is to flag problems (the red squiggles in the report), not to make the model get them right.
- **Misreadings of the data are not caught.** 7B described a steady oxidiser flow as a "sharp pulse"; every number it cited was real and matched its field, so the verifier cannot see it. Before semantic verification, an invented "11 s in total" also passed because it happened to be within 1% of some number in the evidence; closing that loophole is what [semantic verification](#semantic-verification) is for.
- **Small models mostly fail to finish the task, not just invent numbers.** Every 7B run had several tests where it could not submit a report within the step or time limit; 3B mostly submitted empty reports, and a model that says nothing invents nothing. With large run-to-run variation and a small sample, treat these numbers as a rough guide.

**Take these numbers with a grain of salt.** The rule agent was tuned together with this synthetic generator, so a perfect score only shows that the pipeline is self-consistent, not that it handles real data. The benchmark is really for:

1. comparing LLM agents, and how often they invent numbers;
2. catching regressions when tools change;
3. harder scenarios: coupled anomalies, realistic noise spectra, sensor drift, and comparison with real test data.

## Real-data example: HANARO solid motor static fire

`examples/hanaro_knsb/` holds a public KNSB solid-motor static fire by SNU Rocket Team HANARO (2025; data from [snu-hanaro/static-fire-toolkit](https://github.com/snu-hanaro/static-fire-toolkit), MIT License). The original files are kept unchanged in `raw/`, and `prepare.py` turns them into Groundline input:

- Thrust comes from a load cell sampled at irregular times with repeated timestamps. Duplicates are averaged and the signal is interpolated onto a 100 Hz grid; grid points more than 25 ms from any raw sample stay NaN, so dropped frames stay visible.
- The public files do not include the load-cell calibration constants, so volts are converted to newtons with a straight line fitted to HANARO's own processed thrust (R² = 0.9999).
- Pressure comes from a separate logger at about 10 Hz whose clock is not synchronised with the thrust DAQ. The clock offset is the shift that maximises the correlation between pressure and thrust (+5.652 s, correlation 0.9995).
- HANARO does not publish case-pressure or thrust limits, so there are no redlines, and there is no simulation prediction.

```bash
python examples/hanaro_knsb/prepare.py        # regenerate run.csv / meta.json / limits.json
groundline analyze examples/hanaro_knsb/run.csv
```

Rule agent compared with HANARO's own processing:

| | Groundline | HANARO processed |
|---|---|---|
| Peak thrust | 2222.2 N | 2221.7 N |
| Total impulse | 6362 N·s (action time at 10% of peak, 3.91 s)<br>6391 N·s (at 2%, 4.12 s) | 6411 N·s (over its 4.35 s window) |

Because the thrust calibration was fitted to HANARO's curve, matching peaks are expected; the independent checks are the clock alignment, the action time and the impulse.

Two measurement problems were also found, both outside the firing: the thrust channel has 285 data gaps with a median spacing of 0.67 s, regular enough to point to periodic dropped DAQ frames; and there is an isolated thrust spike about 121 s before ignition.

The LLM agent (gpt-5.6-sol) gave 3 findings on this data that agree with the rule agent (sequence, thrust performance, the thrust channel's gaps and spike). All 25 numbers are grounded and the first draft invented none. It also noted on its own that Pc is recorded at only 10 Hz and cannot show oscillations, and checked the thrust channel in the 5–50 Hz band instead (none found). 4 requests, about 13k input tokens.

The data exposed several assumptions built only on the synthetic liquid engine; they are fixed:

- After a slow channel is interpolated onto a fast grid, oscillation detection searched above its Nyquist frequency and spike detection treated every real sample as a spike. Tools now read `native_rate_hz` from `meta.json` and judge at the channel's own rate.
- A quantized, quiet signal (such as steady ambient pressure before ignition) has a median residual of 0, so its noise floor was 0 and a one-step flicker counted as a spike.
- A sensor that does not change before or after the firing is not "frozen"; only flatlines that overlap the firing are reported now.
- Hundreds of recurring dropped frames are reported as one finding instead of hundreds.
- The fast thrust rise at ignition leaked into the lowest bin of the oscillation band and was reported as a "5 Hz oscillation"; windows whose peak sits on that bin are no longer flagged.
- New `pulse_metrics` tool for peak thrust, action time and total impulse.

After the changes, the rule agent still scores 100% recall and precision with 0 false positives on the synthetic benchmark.

## Real-data example: Triton liquid-engine hot fire

`examples/triton_lox/` turns a public hot fire of Triton, a pressure-fed LOX / fuel liquid engine (11 April 2025, data from [aidenmccollum/Triton-Hotfire-Analysis](https://github.com/aidenmccollum/Triton-Hotfire-Analysis)), into Groundline input. That repository has no license, so the data is not copied into Groundline: `prepare.py` downloads it (pinned to a commit) and writes `run.csv` locally, both git-ignored. It is one DAQ log at about 1.94 kHz with 25 channels: three thrust load cells, chamber, manifold, tank and regulator pressures, tank weights and thermocouples. No redlines, valve commands or simulation prediction come with it.

```bash
python examples/triton_lox/prepare.py        # downloads the log, keeps 120-180 s, 2 kHz grid, dropouts left as NaN
groundline analyze examples/triton_lox/run.csv
```

The segmentation agrees with the team's own analysis window (148.35–153.35 s in the log): ignition at 148.41 s, mainstage 148.43–153.16 s, thrust action time 148.41–153.54 s. Peak thrust 1144 lbf, 846 lbf mean over the action time, total impulse 4344 lbf·s (sum of the three load cells, baseline removed); there is no published reference for these values for this run.

It also found measurement problems in the data:

- chamber pressure does not return to baseline after shutdown: it reads about 223 psi to the end of the log, against 13.8 psi before the test (63% of the steady level), while thrust and both manifold pressures are back at ambient;
- the fuel regulator's upstream pressure sits at 5180.25 psi, its maximum, for the whole minute: most likely saturated at full scale;
- all 24 channels drop data at the same moments, 192 gaps of up to 42 ms: dropped DAQ frames;
- spikes on five kinds of sensor within 20 ms at ignition, and a chamber-pressure spike to 467 psi (+20%) lasting about 30 ms half a second later. Groundline reports the coincident spikes as one event without deciding whether they are physical or interference.

This was the first liquid-engine data Groundline saw, and the first rule-agent run produced 131 findings, 87 of them for one stuck sensor. What it exposed, and what changed:

- a stuck value split by every short dropout counted as a new flatline → short gaps inside a flatline no longer split it, and a value stuck at a channel's maximum or minimum is marked as likely saturation;
- gaps shared by every channel were reported once per channel → reported once, as DAQ dropouts;
- a chamber-pressure sensor that sticks high after shutdown went unnoticed and stretched "shutdown" to the end of the log (and, stuck above half the peak, it would be averaged into the steady level) → a no-return-to-baseline check; tail-off ends where the trace settles; the steady level comes from the plateau around the peak;
- the start transient produced "critical" 59 Hz and 134 Hz "oscillations" from single 50 ms windows → an oscillation has to persist over 3 windows; shorter peaks are listed as short events;
- spikes on several kinds of sensor at once were called "measurement glitches, not physical events" → reported as one event that no single sensor explains.

After the changes there are 18 findings, all verified (105 numbers), and the synthetic benchmark and the HANARO results are unchanged (with the new steady-level estimate and, later, baseline-relative thresholds, HANARO's ignition and mainstage boundaries moved by at most 150 ms). A ~126 Hz component shows in Pc, the LOX manifold and thrust throughout the burn; it is real but small (about 0.1% of Pc on average), and Groundline flags the one 0.15 s stretch where it exceeds the default 0.5% criterion, as a warning.

## Real-data example: UVic MULE-1 hybrid motor, checked against the team's own reports

`examples/uvic_mule/` turns four hot fires of MULE-1, UVic Rocketry's N2O / paraffin hybrid motor ([UVicRocketry/Propulsion-Test-Data](https://github.com/UVicRocketry/Propulsion-Test-Data)), into Groundline runs. It is a hybrid, not a liquid engine, but it has a liquid-style oxidiser feed (run tank, flow lines, valve, injector) and ~450 Hz logs, and what makes it valuable is that the team wrote down what happened at every test. The repository has no license, so as with Triton, `prepare.py` downloads the logs and nothing is committed.

```bash
python examples/uvic_mule/prepare.py          # all four hot fires
groundline analyze examples/uvic_mule/2025-01-18/run.csv
```

| Test | The team's report | Groundline (rule agent) |
|---|---|---|
| 2024-12-12 | Igniter wiring shorted, both valves lost power, the DAQ cut out as soon as the engine ignited | "Recording stops during the firing": Pc starts rising at 317.48 s and the log ends at 317.53 s, so no performance is reported. T_post_comb has no signal |
| 2025-01-18 | "First completely nominal hot fire"; thrust ~25% low, cause unclear; chamber thermocouples not installed / damaged | Both chamber thermocouples dead (−273.1 °C: open sensors). Pc reads below vacuum (median −25.4 psi): a zero offset. **P_n2_line and P_run_tank are identical sample for sample**, which the report does not mention: a wiring or configuration error. F/Pc steady within mainstage (9%) but 62% off in the tail-off, with thrust still ~330 N while Pc reads 24–171 psi |
| 2025-02-08 | Nozzle throat insert broke (~7 mm wider), liner shattered, no stable combustion, low thrust | F/Pc changed 79% while burning (1.54 → 2.76 N/psi): the signature of a throat that opened up, or a failing sensor. Pc offset; simultaneous spikes on several channels during start-up |
| 2025-09-20 | No report | Pc saturated at 2021 psi during the firing, so segmentation fell back to thrust (1.16–5.67 s; action time 4.51 s, 1268 N·s). A 10 ms, 2343 N thrust spike at shutdown is recognised as a spike, not the peak. Three thermocouples dead |

What it cannot say: that thrust was 25% below expectation (no prediction is published), what broke on 02-08 beyond "a throat change or a sensor", or whether the 01-18 tail-off mismatch comes from the pressure port or from combustion.

As with Triton, the first run was wrong in instructive ways: phases from a noise blip and from a log cut at ignition, "saturated" for sensors that were simply unplugged, segmentation on a saturated Pc, a 10 ms shutdown slam taken as peak thrust, and a crash on a sub-window oscillation search. What changed, each change general:

- segmentation: thresholds relative to the pre-test baseline (so offsets do not matter), ignition found by searching back from mainstage, a fallback to thrust when Pc shows no clear pulse, and a flag when the log ends during the firing;
- sensor health: one value for the whole record is "no signal", not saturation; physically impossible readings (pressure below vacuum, temperature below absolute zero, for at least 0.5 s); channels with identical samples;
- a new tool, `check_thrust_pressure_ratio`: thrust over chamber pressure is proportional to the thrust coefficient times the throat area, so it should hold while the engine burns; it uses quasi-steady windows only and says whether the change happens within mainstage;
- pulse metrics: the pulse is located on a 0.1 s median, so a short shock cannot pass for the peak.

Regression: the synthetic benchmark is unchanged (100% recall and precision, 134/134 verified), HANARO's thrust results are unchanged (2222.2 N, 6362 / 6391 N·s; ignition and mainstage boundaries moved by at most 150 ms with the baseline-relative thresholds), Triton is unchanged, and the verifier benchmark moved by at most a point because tool results now carry more fields.

## Semantic verification

Checking only that "the number appears in the cited evidence" has two loopholes, and both showed up in the 7B results above: a real number put in the wrong place (every cited number is real, but the meaning is wrong), and a made-up number that happens to match an unrelated number in the evidence within rounding.

So the verifier now remembers which field every evidence number came from (`peak`, `duration_s`, `freq_hz`, ...) and its unit, and reads each number in the sentence together with:

- **its unit**: a number written with s / ms may only match a time field; Hz only a frequency field; % only a percentage field; a physical unit such as bar, K or N·s must match the unit the tool reported. Compound units are read whole and cancelled (kg/s·s = kg). The written unit also fixes the scale: a field of 0.8 s may be written as 800 ms, not as 0.8 ms. A count word ("3 spikes", "3 个") must come from a count field or a list length.
- **its role word**: a number directly introduced by "peak / duration / mean / impulse / frequency / deviation / latency" (or the Chinese equivalents) must come from a field whose name fits that role. The word only counts when it leads straight into the number: in "continuously above the 700 K redline" the word describes the exceedance, and "10% of the peak" is a fraction of the peak; neither triggers the check. Range endpoints (1.27–9.02 s) get no role check either; instead the start of a range takes the unit written after its end (so 1.27 must be a time too), and a time range may not end before it starts.
- **its category**: a finding filed under an anomaly category (redline violation, oscillation, sensor fault, valve response, performance deviation) must cite evidence in which the matching tool actually reported that anomaly, on the finding's channel. A check that passed has to be filed as an observation. This is the mistake behind the 36% precision of the first LLM run above, and re-verifying the stored runs shows it is still the 7B model's most common one: "valve response normal" and "no significant oscillation" filed as anomalies, and plain peak values filed as performance deviations. 5 of its claims are now flagged for it; no gpt-5.6-sol claim and no rule-agent claim is.

This is still deterministic string and field matching; no second LLM is asked to judge.

**Source tags.** Guessing which field a number came from has a ceiling: two numbers that are both "a time" cannot be told apart from the text. So a number may name its field in brackets right after it (and its unit): `742.3 K [E4.violations[0].peak_value]`, `3 spikes [len(E3.issues[0].spikes)]`, `3.86 [E6.events[0].t_start]–6.01 s [E6.events[0].t_end]`. A tagged number is checked against that field only (value, unit, role word, range order); a tag that names the wrong field, or evidence the finding does not cite, is flagged. A bare `[E4]` stays an ordinary reference. In the HTML report the tags become links to their evidence entry. LLM agents are asked to tag every number (`GROUNDLINE_CITE_NUMBERS=0` turns this off); MCP clients can do the same. Untagged numbers are still checked as before.

Unit conversions are also tied to the kind of field now: s ↔ ms only for times, fraction ↔ % only for unitless fractions. A frequency of 802 Hz no longer "matches" a written 8 (×0.01), and a 0.5 % threshold no longer matches "50 %".

**The verifier's own benchmark** (`groundline verifier-bench`): starting from correct rule-agent findings, change one number at a time to plant one of three known errors, and count what the verifier catches: checking only that the value occurs in the cited evidence, the full checks on untagged text, and the full checks when every number carries its source tag. 100 synthetic runs, 50 each in Chinese and English:

| Planted error | Count | Grounding only | Grounding + semantics | With source tags |
|---|---|---|---|---|
| Invented value (changed by −30% to +50%) | 1366 | 89% | **98%** | **100%** |
| Real value in the wrong place (another field of the same evidence) | 1438 | 0% | **67%** | **99.9%** |
| Wrong unit (s ↔ Hz, % → s, ...) | 1170 | 0% | **98%** | **100%** |
| Unchanged correct findings (false alarms) | 268 | 0% | **0%** | **0%** |

Without tags, the 67% for misplaced values splits into two cases: swapping in a field of a different kind (a duration replaced by a peak pressure) is caught 92% of the time; swapping in a field of the same kind (one time replaced by another time) 34%, when the sentence has a role word, the swap breaks a time range, or the s/ms scale no longer fits. (Before the range, scale, count and conversion rules these were 45% overall, 69% and 13%.) That is the limit of reading text alone; with source tags all but one of the 3,974 planted errors are caught (the one left: a swapped-in value that equals the cited field within rounding). Whether LLM agents tag their numbers reliably has not been measured yet: the benchmark runs above predate the tag instruction.

Also note that the role-word rules were written against the rule agent's sentence templates, so the false-alarm rate above, measured on those same templates, is optimistic. On independent text, the two reports written by gpt-5.6-sol (9 findings, 60 numbers, including the HANARO real-data report), the new verifier also raised no false alarms, and re-verifying the stored gpt-5.6-sol and Qwen benchmark runs after the range, scale, count and conversion rules were added (123 gpt-5.6-sol claims, 798 numbers) changed no gpt-5.6-sol verdict. In the 7B drafts it stopped two more numbers, both wrong: a duration of "0.2 s" that only matched a total impulse of 19.4 at ×0.01, and a mean deviation in MPa scaled ×1000 and written as a percentage. A more reliable false-alarm estimate needs more real reports from different models.

LLM benchmark runs store every run's findings together with its evidence ledger. When the verifier improves, `groundline reverify <results.json>` re-scores them without re-running the model; the last 7B run above (`docs/results/leaderboard/`) was re-scored this way.

## Design notes

- **The LLM does no arithmetic.** The system prompt requires every number in a finding to come from evidence. To say "5% deviation", the model has to call `compare_reference` and get that number, not subtract two numbers itself.
- **The rule agent has to pass the verifier too.** During development the verifier caught a hard-coded "5 samples" in a rule template that did not appear in any evidence. The fix was to put the criterion into the tool output, not to loosen the verifier.
- **Findings that attribute causes.** If the flows match the prediction but chamber pressure is low, the report points to low c* efficiency; if a late fuel valve delays ignition, the resulting temperature lag is attributed to the valve delay instead of being reported as a separate problem.

## Layout

```
src/groundline/
  synth.py           synthetic hot-fire data with ground-truth anomalies
  io.py              CSV / TDMS I/O
  session.py         Session and the evidence ledger
  tools.py           deterministic analysis tools (the only source of numbers)
  findings.py        findings and the grounding verifier
  semantics.py       semantic verification: units and role words against evidence fields
  agent.py           rule agent, LLM agent (OpenAI Responses / compatible endpoints / Anthropic)
  report.py          self-contained HTML report and report.json
  reproduce.py       evidence reproduction
  evaluate.py        benchmark and re-verification
  leaderboard.py     multi-model leaderboard
  verifier_bench.py  benchmark of the verifier itself (planted errors)
  hybrid.py          hybrid benchmark: known anomalies injected into real logs
  ingest.py          raw CSV logs to runs, from a (drafted) mapping
  mcp_server.py      MCP server
  config.py          .env loading
  cli.py             command line
```

## Roadmap

- [ ] Validate criteria and calibrate thresholds on more real liquid-engine test data, ideally with redlines, valve commands and a simulation prediction (one public solid-motor and one public liquid-engine example are included)
- [ ] Deviation attribution: when simulation and measurement disagree, list candidate causes (physics, mesh, boundary conditions, manufacturing, sensors) and design checks
- [ ] Cross-run comparison and trend analysis
- [x] LLM agent leaderboard and verifier benchmark
- [ ] Harder synthetic scenarios (coupled anomalies, realistic noise spectra, sensor drift)
- [ ] Test report templates (Word/PDF export)

## License

MIT
