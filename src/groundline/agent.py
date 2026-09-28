"""Planners that turn a Session into findings.

* :class:`RuleAgent` — a fixed, deterministic playbook (no LLM).  It is the
  baseline in the benchmark and the fallback when no model is configured.
* :class:`LLMAgent` — an LLM chooses which tools to call, drills down, and
  writes the findings.  After it submits, the verifier checks every claim; if
  something is ungrounded the agent gets one chance to fix it.

Both produce the same :class:`AnalysisResult`, rendered by ``groundline.report``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .findings import CATEGORIES, SEVERITIES, Finding, verify_findings
from .session import Session, to_jsonable
from .tools import REGISTRY


@dataclass
class AnalysisResult:
    findings: list[Finding]
    summary: str
    verification: dict
    agent: dict
    transcript: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0


# =============================================================================
# Rule-based playbook
# =============================================================================
_TXT = {
    "zh": {
        "phase_t": "试验时序：点火 {ign:.3f} s，主级 {ms0:.3f}–{ms1:.3f} s",
        "phase_s": "基于 {ch} 分段：点火 {ign:.3f} s，主级段 {ms0:.3f}–{ms1:.3f} s（持续 {dur:.2f} s），"
                   "稳态水平 {lvl:.3f} {unit}。",
        "osc_t": "{ch} 出现 {f:.0f} Hz 窄带振荡",
        "osc_s": "{ch} 在 {t0:.2f}–{t1:.2f} s 检测到约 {f:.0f} Hz 的窄带振荡，峰值幅值为均值的 {amp:.2f}%"
                 "（判据 {thr:.1f}%）。",
        "osc_corr": " {ch2} 在 {f2:.0f} Hz 同步检出振荡，佐证其为物理现象而非单个传感器问题。",
        "red_t": "{ch} 超出红线（{kind} {lim:g} {unit}）",
        "red_s": "{ch} 在 {t0:.3f}–{t1:.3f} s 持续超出红线 {lim:g} {unit}，持续 {dur:.3f} s，峰值 {pk:.2f} {unit}"
                 "（{tp:.3f} s）。",
        "pulse_t": "{ch} 峰值 {pk:.0f} {unit}，工作时间 {dur:.2f} s，总冲 {imp:.0f} {iu}",
        "pulse_s": "按峰值的 {p0:g}% 截取工作时间 {t0:.2f}–{t1:.2f} s（{dur:.2f} s）；{ch} 在 {tp:.2f} s 达到峰值 "
                   "{pk:.1f} {unit}，工作时间内平均 {mean:.1f} {unit}，积分 {imp:.1f} {iu}（已扣除基线 {base:.1f} {unit}）。",
        "rec_t": "{ch} 反复出现数据缺失",
        "rec_s": "{ch} 在 {t0:.2f}–{t1:.2f} s 内共有 {n} 段数据缺失，合计 {tot:.2f} s，最长 {lg:.3f} s，"
                 "相邻两段的间隔中位数 {iv:.2f} s；其中 {nf} 段落在点火到拖尾结束之间。缺失有规律地重复出现，"
                 "更像采集系统丢帧，而不是被测量本身的变化。",
        "nan_t": "{ch} 数据缺失",
        "nan_s": "{ch} 在 {t0:.3f}–{t1:.3f} s 出现 NaN 数据缺失，共 {n} 个采样点（{dur:.3f} s）。",
        "flat_t": "{ch} 信号冻结",
        "flat_s": "{ch} 在 {t0:.3f}–{t1:.3f} s 数值恒定为 {v:.4g}（{dur:.3f} s），疑似传感器或采集通道故障。",
        "spk_t": "{ch} 出现 {n} 个孤立尖峰",
        "spk_s": "{ch} 在 {t0:.3f}–{t1:.3f} s 之间出现 {n} 个孤立尖峰（宽度不超过 {w} 个采样点），"
                 "持续时间短于红线判据，判断为测量毛刺，而非真实的物理变化。",
        "valve_t": "{cmd} 开启响应延迟 {lat:.1f} ms",
        "valve_s": "{cmd} 在 {tc:.3f} s 发出开启指令后，{resp} 在 {lat:.1f} ms 后才开始响应，超过允许的 {lim:.0f} ms。",
        "valve_cons": " 点火随之推迟，{ch} 在 {t0:.2f}–{t1:.2f} s 相对预测偏差 {dev:.2f}%，为该延迟的后果而非独立问题。",
        "dev_t": "{ch} 与仿真预测偏差 {dev:.1f}%",
        "dev_s": "{ch} 在 {t0:.2f}–{t1:.2f} s 持续偏离仿真预测，平均偏差 {dev:.2f}%（容差 {tol:.1f}%）。",
        "dev_inj": " 喷前压力同步偏低（{chs}），与室压变化一致。",
        "dev_cstar": " 同期 {flows} 与预测一致（平均偏差 {fdev}），流量正常而室压偏低，指向燃烧效率（c*）不足，"
                     "建议检查喷注器与混合比。",
        "summary": "共分析 {n_ev} 项证据，得出 {n_f} 条结论，其中关键 {n_c} 条、警告 {n_w} 条。",
        "clean": "未发现超出判据的异常。",
    },
    "en": {
        "phase_t": "Test sequence: ignition {ign:.3f} s, mainstage {ms0:.3f}–{ms1:.3f} s",
        "phase_s": "Segmented on {ch}: ignition at {ign:.3f} s, mainstage {ms0:.3f}–{ms1:.3f} s ({dur:.2f} s), "
                   "steady level {lvl:.3f} {unit}.",
        "osc_t": "{f:.0f} Hz narrow-band oscillation on {ch}",
        "osc_s": "{ch} shows a narrow-band oscillation at about {f:.0f} Hz between {t0:.2f} and {t1:.2f} s; "
                 "peak amplitude {amp:.2f}% of mean (criterion {thr:.1f}%).",
        "osc_corr": " {ch2} shows the same oscillation at {f2:.0f} Hz, so it is physical rather than a single-sensor artefact.",
        "red_t": "{ch} exceeded its redline ({kind} {lim:g} {unit})",
        "red_s": "{ch} stayed beyond its {lim:g} {unit} redline from {t0:.3f} to {t1:.3f} s ({dur:.3f} s), "
                 "peaking at {pk:.2f} {unit} at {tp:.3f} s.",
        "pulse_t": "{ch} peak {pk:.0f} {unit}, action time {dur:.2f} s, total impulse {imp:.0f} {iu}",
        "pulse_s": "Action time taken at {p0:g}% of peak: {t0:.2f}–{t1:.2f} s ({dur:.2f} s). {ch} peaks at {pk:.1f} {unit} "
                   "at {tp:.2f} s, averages {mean:.1f} {unit} over the action time, and integrates to {imp:.1f} {iu} "
                   "(baseline {base:.1f} {unit} removed).",
        "rec_t": "{ch} has recurring data gaps",
        "rec_s": "{ch} has {n} data gaps between {t0:.2f} and {t1:.2f} s, {tot:.2f} s in total, the longest "
                 "{lg:.3f} s, with a median spacing of {iv:.2f} s; {nf} of them fall between ignition and the end of "
                 "tail-off. Gaps that repeat this regularly point to dropped DAQ frames rather than to the measured "
                 "quantity.",
        "nan_t": "{ch} data gap",
        "nan_s": "{ch} returned NaN from {t0:.3f} to {t1:.3f} s ({n} samples, {dur:.3f} s).",
        "flat_t": "{ch} signal frozen",
        "flat_s": "{ch} was stuck at {v:.4g} from {t0:.3f} to {t1:.3f} s ({dur:.3f} s): likely a sensor or DAQ fault.",
        "spk_t": "{n} isolated spikes on {ch}",
        "spk_s": "{ch} has {n} isolated spikes between {t0:.3f} and {t1:.3f} s (at most {w} samples wide, shorter than the "
                 "redline persistence): measurement glitches, not physical events.",
        "valve_t": "{cmd} opening latency {lat:.1f} ms",
        "valve_s": "After the {cmd} open command at {tc:.3f} s, {resp} only responded after {lat:.1f} ms, "
                   "beyond the {lim:.0f} ms limit.",
        "valve_cons": " Ignition was delayed accordingly: {ch} deviates {dev:.2f}% from prediction from {t0:.2f} to "
                      "{t1:.2f} s, a consequence of this delay rather than a separate problem.",
        "dev_t": "{ch} deviates {dev:.1f}% from prediction",
        "dev_s": "{ch} deviates from the simulation prediction from {t0:.2f} to {t1:.2f} s, mean deviation {dev:.2f}% "
                 "(tolerance {tol:.1f}%).",
        "dev_inj": " Injector pressures shift with it ({chs}), consistent with the chamber pressure change.",
        "dev_cstar": " Over the same window {flows} match the prediction (mean deviation {fdev}); nominal flow with low "
                     "chamber pressure points to low combustion efficiency (c*) — check injector and mixture ratio.",
        "summary": "{n_ev} pieces of evidence, {n_f} findings: {n_c} critical, {n_w} warnings.",
        "clean": "No anomaly beyond the criteria was found.",
    },
}


class RuleAgent:
    name = "rule"

    def __init__(self, lang: str = "zh"):
        self.lang = lang if lang in _TXT else "zh"

    def run(self, s: Session) -> AnalysisResult:
        t0 = time.perf_counter()
        T = _TXT[self.lang]
        F: list[Finding] = []
        s.run("describe_data")
        seg = s.run("segment_phases")
        r = seg.result
        if r.get("fired"):
            F.append(Finding(
                T["phase_t"].format(ign=r["ignition_s"], ms0=r["mainstage_start_s"], ms1=r["mainstage_end_s"]),
                T["phase_s"].format(ch=r["channel"], ign=r["ignition_s"], ms0=r["mainstage_start_s"],
                                    ms1=r["mainstage_end_s"], dur=r["mainstage_duration_s"],
                                    lvl=r["steady_level"], unit=r["unit"]),
                "observation", "info", r["channel"], r["mainstage_start_s"], r["mainstage_end_s"], [seg.id]))

        for fc in s.channels_of_kind("force"):
            pm = s.run("pulse_metrics", channel=fc)
            p = pm.result
            kw = dict(ch=fc, pk=p["peak"], unit=p["unit"], dur=p["action_time_s"], imp=p["integral"],
                      iu=p["integral_unit"], p0=p["start_pct"], t0=p["t_start"], t1=p["t_end"], tp=p["t_peak"],
                      mean=p["mean_over_action_time"], base=p["baseline"])
            F.append(Finding(T["pulse_t"].format(**kw), T["pulse_s"].format(**kw), "observation", "info", fc,
                             p["t_start"], p["t_end"], [pm.id]))

        covered: set[str] = set()  # channels already explained by a finding

        health = s.run("check_sensor_health")
        for i in health.result["issues"]:
            ch = i["channel"]
            if i["kind"] == "recurring_nan_gaps":
                t, st = T["rec_t"], T["rec_s"].format(ch=ch, t0=i["t_start"], t1=i["t_end"], n=i["count"],
                                                     tot=i["total_s"], lg=i["longest_s"], iv=i["median_interval_s"],
                                                     nf=i["count_during_firing"])
            elif i["kind"] == "nan_gap":
                t, st = T["nan_t"], T["nan_s"].format(ch=ch, t0=i["t_start"], t1=i["t_end"], n=i["n_samples"],
                                                     dur=i["duration_s"])
            elif i["kind"] == "flatline":
                t, st = T["flat_t"], T["flat_s"].format(ch=ch, t0=i["t_start"], t1=i["t_end"], v=i["stuck_value"],
                                                       dur=i["duration_s"])
            else:
                t, st = T["spk_t"], T["spk_s"].format(ch=ch, t0=i["t_start"], t1=i["t_end"], n=i["count"],
                                                     w=health.result["criteria"]["spike_max_width_samples"])
            F.append(Finding(t.format(ch=ch, n=i.get("count", "")), st, "sensor_fault", "warning", ch,
                             i["t_start"], i["t_end"], [health.id]))
            covered.add(ch)

        red = s.run("check_redlines")
        for v in red.result.get("violations", []):
            F.append(Finding(
                T["red_t"].format(ch=v["channel"], kind=v["kind"], lim=v["limit"], unit=v["unit"]),
                T["red_s"].format(ch=v["channel"], t0=v["t_start"], t1=v["t_end"], lim=v["limit"], unit=v["unit"],
                                  dur=v["duration_s"], pk=v["peak_value"], tp=v["t_peak"]),
                "redline_violation", "critical", v["channel"], v["t_start"], v["t_end"], [red.id]))
            covered.add(v["channel"])

        valves = s.run("measure_valve_response")
        for e in valves.result["events"]:
            if e["exceeds_limit"] and e["latency_ms"] is not None:
                F.append(Finding(
                    T["valve_t"].format(cmd=e["command"], lat=e["latency_ms"]),
                    T["valve_s"].format(cmd=e["command"], tc=e["t_cmd"], resp=e["response"], lat=e["latency_ms"],
                                        lim=e["limit_ms"]),
                    "valve_response", "warning", e["response"], e["t_cmd"], e["t_response"], [valves.id]))

        pc = seg.result.get("channel", "Pc")
        osc = s.run("detect_oscillation", channel=pc)
        for e in osc.result["events"]:
            ids = [osc.id]
            corr = ""
            for ch2 in [c for c in s.channels_of_kind("vibration")] + [
                c for c in s.channels_of_kind("pressure") if c != pc
            ]:
                o2 = s.run("detect_oscillation", channel=ch2, t_start=e["t_start"], t_end=e["t_end"])
                hits = [x for x in o2.result["events"] if abs(x["freq_hz"] - e["freq_hz"]) < 0.05 * e["freq_hz"]]
                if hits:
                    ids.append(o2.id)
                    corr = T["osc_corr"].format(ch2=ch2, f2=hits[0]["freq_hz"])
                    break
            amp = e.get("peak_amp_pct", 0.0)
            F.append(Finding(
                T["osc_t"].format(ch=pc, f=e["freq_hz"]),
                T["osc_s"].format(ch=pc, t0=e["t_start"], t1=e["t_end"], f=e["freq_hz"], amp=amp,
                                  thr=osc.result["threshold_pct"]) + corr,
                "combustion_oscillation", "critical" if amp >= 2 * osc.result["threshold_pct"] else "warning",
                pc, e["t_start"], e["t_end"], ids))

        if s.reference is not None:
            ref_ch = [c for c in s.reference.columns if c != "time" and c in s.channels]
            cmp = {c: s.run("compare_reference", channel=c) for c in ref_ch}
            flows = [c for c in ref_ch if s.channel_info(c).get("kind") == "flow"]
            # injector pressures sit on top of Pc: if Pc deviates, fold their deviation into the Pc finding
            pc_iv = cmp[pc].result.get("sustained_deviation_intervals", []) if pc in cmp else []
            folded = {}
            for c, ev in cmp.items():
                if c == pc or s.channel_info(c).get("kind") != "pressure" or c in covered:
                    continue
                ivs = ev.result.get("sustained_deviation_intervals", [])
                if pc_iv and ivs and all(any(i["t_start"] < j["t_end"] and j["t_start"] < i["t_end"] for j in pc_iv)
                                         for i in ivs):
                    folded[c] = ev
            for c, ev in cmp.items():
                res = ev.result
                if c in covered or c in flows or c in folded or res.get("error"):
                    continue
                for iv in res["sustained_deviation_intervals"]:
                    ids = [ev.id]
                    st = T["dev_s"].format(ch=c, t0=iv["t_start"], t1=iv["t_end"], dev=iv["mean_dev_pct"],
                                           tol=res["tolerance_pct"])
                    if c == pc and folded:
                        st += T["dev_inj"].format(chs=", ".join(
                            f"{fc} {fe.result['sustained_deviation_intervals'][0]['mean_dev_pct']:.2f}%"
                            for fc, fe in folded.items()))
                        ids += [fe.id for fe in folded.values()]
                    if c == pc and flows and all(cmp[fc].result.get("within_tolerance") for fc in flows):
                        fdev = ", ".join(f"{cmp[fc].result['mean_dev_pct']:.2f}%" for fc in flows)
                        st += T["dev_cstar"].format(flows=", ".join(flows), fdev=fdev)
                        ids += [cmp[fc].id for fc in flows]
                    F.append(Finding(T["dev_t"].format(ch=c, dev=abs(iv["mean_dev_pct"])), st,
                                     "performance_deviation", "warning", c, iv["t_start"], iv["t_end"], ids))
            for fc in flows:
                res = cmp[fc].result
                if fc not in covered and not res.get("error"):
                    for iv in res["sustained_deviation_intervals"]:
                        F.append(Finding(T["dev_t"].format(ch=fc, dev=abs(iv["mean_dev_pct"])),
                                         T["dev_s"].format(ch=fc, t0=iv["t_start"], t1=iv["t_end"],
                                                           dev=iv["mean_dev_pct"], tol=res["tolerance_pct"]),
                                         "performance_deviation", "warning", fc, iv["t_start"], iv["t_end"],
                                         [cmp[fc].id]))

        # a late valve delays ignition: early-mainstage deviations on other channels are its consequence
        valve_f = [f for f in F if f.category == "valve_response"]
        ms0 = seg.result.get("mainstage_start_s")
        if valve_f and ms0 is not None:
            for f in [f for f in F if f.category == "performance_deviation" and f.channel != pc
                      and f.t_start is not None and f.t_start - ms0 < 0.6]:
                F.remove(f)
                ev = s.evidence(f.evidence[0])
                iv = next(i for i in ev.result["sustained_deviation_intervals"] if i["t_start"] == f.t_start)
                valve_f[0].statement += T["valve_cons"].format(ch=f.channel, t0=iv["t_start"], t1=iv["t_end"],
                                                               dev=iv["mean_dev_pct"])
                valve_f[0].evidence += [e for e in f.evidence if e not in valve_f[0].evidence]

        order = {k: i for i, k in enumerate(SEVERITIES)}
        F.sort(key=lambda f: (order.get(f.severity, 9), f.t_start if f.t_start is not None else 0))
        ver = verify_findings(F, s)
        n_c = sum(f.severity == "critical" for f in F)
        n_w = sum(f.severity == "warning" for f in F)
        summary = T["summary"].format(n_ev=len(s.ledger), n_f=len(F), n_c=n_c, n_w=n_w)
        if not (n_c or n_w):
            summary += " " + T["clean"]
        return AnalysisResult(F, summary, ver, {"type": "rule", "lang": self.lang},
                              elapsed_s=time.perf_counter() - t0)


# =============================================================================
# LLM agent
# =============================================================================
class Backend(Protocol):
    name: str
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        """Return {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}."""


SUBMIT_TOOL = {
    "name": "submit_report",
    "description": "Submit the final analysis. Call exactly once when done.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "2-4 sentence executive summary"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "statement": {"type": "string",
                                      "description": "the claim; every number must come from the cited evidence"},
                        "category": {"type": "string", "enum": list(CATEGORIES),
                                     "description": "the anomaly class found; use 'observation' for "
                                                    "checks that passed or found nothing"},
                        "severity": {"type": "string", "enum": list(SEVERITIES)},
                        "channel": {"type": "string"},
                        "t_start": {"type": "number"},
                        "t_end": {"type": "number"},
                        "evidence": {"type": "array", "items": {"type": "string"},
                                     "description": "evidence IDs such as E3"},
                    },
                    "required": ["title", "statement", "category", "severity", "evidence"],
                },
            },
        },
        "required": ["summary", "findings"],
    },
}

SYSTEM_PROMPT = """You are a test engineer analysing data from a rocket engine or motor firing test (liquid engine
hot-fire, solid or hybrid motor static fire, or a similar bench test).

You cannot see raw samples. You work only through analysis tools; every tool call is stored as an evidence entry
with an ID (E1, E2, ...). Your report is checked by a verifier:
- every finding must cite the evidence IDs it relies on;
- every number you write must appear in the cited evidence results (rounding is fine; you may convert s<->ms or
  fraction<->%). Do not compute new numbers yourself (no subtraction, ratios or averages of your own) — if you need
  a number, call a tool that produces it;
- t_start/t_end of a finding must come from the evidence as well.

Method:
1. Start from the overview and phase segmentation you are given.
2. Check sensor health first, so you do not mistake instrumentation faults for engine behaviour.
3. Check redlines, valve response, oscillations (on the chamber pressure and then on other channels to corroborate),
   and compare with the simulation prediction for each channel that has one.
4. When something is found, drill down (other channels, narrower windows) to explain it; when a symptom is explained
   by another (e.g. a flow deviation caused by a sensor dropout) say so instead of reporting it twice.
5. Not every test has valves, a simulation prediction or redlines, and some channels are recorded slower than the
   grid (see native_rate_hz in the overview). Skip checks that do not apply and mention that in the summary instead
   of writing findings about them. For thrust or other pulse-shaped channels use pulse_metrics (peak, action time,
   total impulse).
6. Include one 'observation' finding describing the test sequence.
7. Call submit_report once. Categories: {categories}. Severities: critical (safety/redline/instability),
   warning (needs engineering attention), info.
   A category other than 'observation' means "this anomaly was found". A check that passed or found nothing
   ("no redline exceeded", "valve response within limit", "no oscillation detected", "matches the simulation")
   is NOT an anomaly: use category 'observation' and severity 'info' for it, or fold it into the summary.
   Likewise a deviation that is a consequence of an anomaly you already report (e.g. a temperature lag caused by
   a late valve opening) belongs in that anomaly's finding, not in a finding of its own.
Write titles and statements in {language}. Be concise and specific; engineers will read this.
"""


def _tool_specs() -> list[dict]:
    specs = [{"name": t.name, "description": t.description, "parameters": t.json_schema()}
             for t in REGISTRY.values()]
    return specs + [SUBMIT_TOOL]


def _compact(result: dict, limit: int = 6000) -> str:
    txt = json.dumps(result, ensure_ascii=False)
    if len(txt) <= limit:
        return txt
    r = dict(result)
    for k, v in list(r.items()):
        if isinstance(v, list) and len(v) > 10:
            r[k] = v[:10] + [f"... {len(v) - 10} more"]
    txt = json.dumps(r, ensure_ascii=False)
    return txt[:limit] + ("..." if len(txt) > limit else "")


class LLMAgent:
    def __init__(self, backend: Backend, lang: str = "zh", max_steps: int = 24, fix_rounds: int = 1,
                 prefetch: bool = True):
        self.backend = backend
        self.lang = lang
        self.max_steps = max_steps
        self.fix_rounds = fix_rounds
        self.prefetch = prefetch

    def run(self, s: Session) -> AnalysisResult:
        t0 = time.perf_counter()
        system = SYSTEM_PROMPT.format(categories=", ".join(CATEGORIES),
                                      language="Simplified Chinese" if self.lang == "zh" else "English")
        tools = _tool_specs()
        intro = ["Analyse this test run and submit a report."]
        if self.prefetch:
            for name in ("describe_data", "segment_phases"):
                ev = s.run(name)
                intro.append(f"[{ev.id}] {name}: {_compact(ev.result)}")
        messages: list[dict] = [{"role": "user", "content": "\n\n".join(intro)}]
        transcript: list[dict] = []
        findings: list[Finding] | None = None
        summary = ""
        fixes_left = self.fix_rounds
        ver: dict = {}
        first_ver: dict | None = None
        first_flagged: list[dict] = []

        for _step in range(self.max_steps):
            reply = self.backend.complete(system, messages, tools)
            calls = reply.get("tool_calls") or []
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
            transcript.append({"role": "assistant", "content": reply.get("content"),
                               "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls]})
            if not calls:
                messages.append({"role": "user", "content": "Continue with tool calls, then call submit_report."})
                continue
            submitted = False
            for c in calls:
                name, args = c["name"], c.get("arguments") or {}
                parsed = None
                if "__invalid_json__" in args:
                    parsed = "arguments were not valid JSON: " + args["__invalid_json__"][:200]
                elif name == "submit_report":
                    try:
                        raw_f = args.get("findings", [])
                        if isinstance(raw_f, str):  # some models send the list as a JSON string
                            raw_f = json.loads(raw_f)
                        new_findings = [Finding.from_dict(d) for d in raw_f]
                    except Exception as e:
                        parsed = f"could not read findings ({type(e).__name__}: {e})"[:300]
                if parsed:
                    content = json.dumps({"error": parsed + ". Call the tool again with valid arguments."})
                    messages.append({"role": "tool", "tool_call_id": c["id"], "name": name, "content": content})
                    transcript.append({"role": "tool", "name": name, "content": content})
                    continue
                if name == "submit_report":
                    findings = new_findings
                    summary = args.get("summary", "")
                    ver = verify_findings(findings, s)
                    if first_ver is None:
                        first_ver = dict(ver)
                        first_flagged = [{"title": f.title, "statement": f.statement, "evidence": f.evidence,
                                          "ungrounded_numbers": f.verification["ungrounded_numbers"],
                                          "problems": f.verification["problems"]}
                                         for f in findings if f.verification["status"] != "verified"]
                    bad = [(i, f) for i, f in enumerate(findings) if f.verification["status"] != "verified"]
                    if bad and fixes_left > 0:
                        fixes_left -= 1
                        lines = [f"finding {i} ({f.title}): ungrounded numbers {f.verification['ungrounded_numbers']}; "
                                 f"problems {f.verification['problems']}" for i, f in bad]
                        content = ("Verifier rejected some claims:\n" + "\n".join(lines) +
                                   "\nFix them (cite the right evidence, call tools for missing numbers, or remove "
                                   "the number) and call submit_report again with the full list.")
                    else:
                        submitted = True
                        content = json.dumps({"accepted": True, "verification": ver})
                else:
                    try:
                        ev = s.run(name, **args)
                        content = json.dumps({"evidence_id": ev.id, "result": json.loads(_compact(ev.result))}
                                             if len(json.dumps(ev.result)) < 6000 else
                                             {"evidence_id": ev.id, "result_truncated": _compact(ev.result)},
                                             ensure_ascii=False)
                    except Exception as e:  # tool errors go back to the model
                        content = json.dumps({"error": f"{type(e).__name__}: {e}"})
                messages.append({"role": "tool", "tool_call_id": c["id"], "name": name, "content": content})
                transcript.append({"role": "tool", "name": name, "content": content[:2000]})
            if submitted:
                break
        if findings is None:
            findings, summary = [], "Agent stopped without submitting a report."
            ver = verify_findings(findings, s)
        return AnalysisResult(findings, summary, ver,
                              {"type": "llm", "backend": self.backend.name, "model": self.backend.model,
                               "lang": self.lang, "first_submission": first_ver, "submitted": first_ver is not None,
                               "first_draft_flagged": first_flagged,
                               "fix_rounds_used": self.fix_rounds - fixes_left,
                               "usage": dict(getattr(self.backend, "usage", {}) or {})},
                              transcript, elapsed_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------- backends
def _parse_args(raw) -> dict:
    """Tool-call arguments as a dict. Unparseable JSON is kept so the agent can tell the model,
    instead of silently running the tool (or accepting a report) with no arguments."""
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__invalid_json__": str(raw)[:500]}
    return v if isinstance(v, dict) else {"__invalid_json__": str(raw)[:500]}


def _check(r) -> None:
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API error {r.status_code}: {r.text[:800]}")


def _post(url: str, body: dict, headers: dict, timeout: float, retries: int = 3):
    """POST with a small retry on rate limits / transient server errors."""
    import httpx

    for attempt in range(retries + 1):
        r = httpx.post(url, json=body, headers=headers, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(min(2 ** attempt * 2, 20))
            continue
        _check(r)
        return r


class OpenAIResponses:
    """OpenAI Responses API (/v1/responses) — needed for tool calling with reasoning models such as gpt-5.6-sol.

    Conversation state is kept server-side with ``previous_response_id``; each call only sends what is new
    since the last one (tool outputs, user nudges).
    """

    name = "openai-responses"

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 reasoning_effort: str | None = None, timeout: float = 300.0):
        self.model = model or os.environ.get("GROUNDLINE_LLM_MODEL", "gpt-5.6-sol")
        self.base_url = (base_url or os.environ.get("GROUNDLINE_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("GROUNDLINE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.reasoning_effort = reasoning_effort or os.environ.get("GROUNDLINE_LLM_REASONING_EFFORT") or None
        self.timeout = timeout
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        self._prev_id: str | None = None
        self._sent = 0  # number of agent messages already delivered to the server

    @staticmethod
    def _to_items(messages: list[dict]) -> list[dict]:
        items: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                items.append({"role": "user", "content": m["content"]})
            elif m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
            # assistant turns already live on the server via previous_response_id
        return items

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        if self._prev_id is None or len(messages) < self._sent:
            self._prev_id, self._sent = None, 0
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": self._to_items(messages[self._sent:]),
            "tools": [{"type": "function", "name": t["name"], "description": t["description"],
                       "parameters": t["parameters"], "strict": False} for t in tools],
        }
        if self._prev_id:
            body["previous_response_id"] = self._prev_id
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = _post(f"{self.base_url}/responses", body, headers, self.timeout)
        data = r.json()
        u = data.get("usage") or {}
        self.usage["input_tokens"] += u.get("input_tokens", 0)
        self.usage["output_tokens"] += u.get("output_tokens", 0)
        self.usage["requests"] += 1
        self._prev_id = data.get("id")
        # the assistant message the agent appends next is represented server-side; skip it next time
        self._sent = len(messages) + 1
        text, calls = [], []
        for item in data.get("output", []):
            if item.get("type") == "message":
                text += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
            elif item.get("type") == "function_call":
                args = _parse_args(item.get("arguments"))
                calls.append({"id": item["call_id"], "name": item["name"], "arguments": args})
        return {"content": "".join(text), "tool_calls": calls}


class OpenAICompatible:
    """Any OpenAI-compatible /chat/completions endpoint: OpenAI, DeepSeek, Qwen (DashScope), vLLM, Ollama..."""

    name = "openai-compatible"

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float | None = None, timeout: float = 180.0):
        self.model = model or os.environ.get("GROUNDLINE_LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("GROUNDLINE_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("GROUNDLINE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        t = os.environ.get("GROUNDLINE_LLM_TEMPERATURE")
        self.temperature = temperature if temperature is not None else (float(t) if t else None)
        self.reasoning_effort = os.environ.get("GROUNDLINE_LLM_REASONING_EFFORT") or None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        import httpx

        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                mm: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    mm["tool_calls"] = [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                        for c in m["tool_calls"]]
                msgs.append(mm)
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        body = {
            "model": self.model,
            "messages": msgs,
            "tools": [{"type": "function", "function": t} for t in tools],
        }
        if self.temperature is not None:  # reasoning models reject a non-default temperature
            body["temperature"] = self.temperature
        if self.reasoning_effort:  # e.g. none | low | medium | high for reasoning models
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = _post(f"{self.base_url}/chat/completions", body, headers, self.timeout)
        data = r.json()
        u = data.get("usage") or {}
        self.usage["input_tokens"] += u.get("prompt_tokens", 0)
        self.usage["output_tokens"] += u.get("completion_tokens", 0)
        self.usage["requests"] += 1
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            args = _parse_args(c["function"].get("arguments"))
            calls.append({"id": c.get("id") or f"call_{len(calls)}", "name": c["function"]["name"], "arguments": args})
        return {"content": msg.get("content") or "", "tool_calls": calls}


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 4096,
                 timeout: float = 180.0):
        self.model = model or os.environ.get("GROUNDLINE_LLM_MODEL", "claude-sonnet-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        import httpx

        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "..."}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) and \
                        out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": "user", "content": m["content"]})
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": out,
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in tools],
        }
        r = httpx.post("https://api.anthropic.com/v1/messages", json=body, timeout=self.timeout,
                       headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        _check(r)
        data = r.json()
        u = data.get("usage") or {}
        self.usage["input_tokens"] += u.get("input_tokens", 0)
        self.usage["output_tokens"] += u.get("output_tokens", 0)
        self.usage["requests"] += 1
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        calls = [{"id": b["id"], "name": b["name"], "arguments": b.get("input") or {}}
                 for b in data["content"] if b["type"] == "tool_use"]
        return {"content": text, "tool_calls": calls}


class ScriptedBackend:
    """Replays a fixed list of replies — for tests and offline demos."""

    name = "scripted"

    def __init__(self, replies: list[dict], model: str = "scripted"):
        self.replies = list(replies)
        self.model = model
        self.calls = 0

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        self.calls += 1
        if not self.replies:
            return {"content": "", "tool_calls": []}
        r = self.replies.pop(0)
        return r(messages) if callable(r) else r


def make_agent(kind: str = "rule", lang: str = "zh", model: str | None = None, base_url: str | None = None,
               api_key: str | None = None):
    if kind == "rule":
        return RuleAgent(lang)
    if kind in ("openai", "openai-compatible", "qwen", "deepseek", "ollama", "vllm"):
        # official OpenAI -> Responses API (tools + reasoning); other endpoints -> Chat Completions.
        # Override with GROUNDLINE_OPENAI_API=responses|chat.
        url = base_url or os.environ.get("GROUNDLINE_LLM_BASE_URL", "https://api.openai.com/v1")
        api = os.environ.get("GROUNDLINE_OPENAI_API") or ("responses" if "api.openai.com" in url else "chat")
        if kind == "openai" and api == "responses":
            return LLMAgent(OpenAIResponses(model, base_url, api_key), lang)
        return LLMAgent(OpenAICompatible(model, base_url, api_key), lang)
    if kind == "anthropic":
        return LLMAgent(AnthropicBackend(model, api_key), lang)
    raise ValueError(f"unknown agent {kind!r}")


__all__ = ["AnalysisResult", "RuleAgent", "LLMAgent", "OpenAICompatible", "OpenAIResponses", "AnthropicBackend", "ScriptedBackend",
           "make_agent", "to_jsonable"]
