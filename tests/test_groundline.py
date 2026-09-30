import json

import numpy as np
import pytest

from groundline import Session
from groundline.agent import LLMAgent, RuleAgent, ScriptedBackend
from groundline.evaluate import match, run_benchmark
from groundline.findings import Finding, extract_numbers, verify_findings
from groundline.report import write_report
from groundline.reproduce import reproduce
from groundline.synth import ANOMALY_TYPES, generate_run


def session_for(anomalies, seed=3):
    r = generate_run(seed, anomalies)
    return r, Session(r.data, r.meta, r.reference, r.limits)


def test_synth_is_deterministic():
    a = generate_run(11, ["oscillation"])
    b = generate_run(11, ["oscillation"])
    assert np.allclose(a.data.to_numpy(), b.data.to_numpy())
    assert a.truth[0].params == b.truth[0].params


def test_synth_rejects_unknown_anomaly():
    with pytest.raises(ValueError):
        generate_run(0, ["banana"])


def test_phases_on_nominal_run():
    _, s = session_for([])
    r = s.run("segment_phases").result
    assert r["fired"]
    assert 1.0 < r["ignition_s"] < 1.2
    assert 8.9 < r["mainstage_end_s"] < 9.1


def test_oscillation_frequency_and_window():
    run, s = session_for(["oscillation"], seed=21)
    truth = run.truth[0]
    ev = s.run("detect_oscillation", channel="Pc").result
    assert ev["detected"]
    e = ev["events"][0]
    assert abs(e["freq_hz"] - truth.params["freq_hz"]) < 3
    assert abs(e["t_start"] - truth.t_start) < 0.15
    assert abs(e["peak_amp_pct"] - truth.params["amplitude_pct"]) < 0.6


def test_nominal_run_has_no_oscillation_or_redline():
    _, s = session_for([], seed=5)
    assert not s.run("detect_oscillation").result["detected"]
    assert s.run("check_redlines").result["n_violations"] == 0
    assert s.run("check_sensor_health").result["n_issues"] == 0


def test_spikes_are_sensor_faults_not_redlines():
    run, s = session_for(["sensor_spike"], seed=8)
    health = s.run("check_sensor_health").result
    spikes = [i for i in health["issues"] if i["kind"] == "spike"]
    assert spikes and spikes[0]["count"] == run.truth[0].params["count"]
    red = s.run("check_redlines").result
    assert red["n_violations"] == 0


def test_valve_delay_measured():
    run, s = session_for(["valve_delay"], seed=4)
    extra = run.truth[0].params["extra_delay_ms"]
    ev = s.run("measure_valve_response").result["events"]
    fu = next(e for e in ev if e["command"] == "cmd_fu" and e["edge"] == "open")
    ox = next(e for e in ev if e["command"] == "cmd_ox" and e["edge"] == "open")
    assert fu["exceeds_limit"] and not ox["exceeds_limit"]
    assert abs((fu["latency_ms"] - ox["latency_ms"]) - extra) < 10


def test_extract_numbers_skips_identifiers():
    nums = [v for _, v, _ in extract_numbers("E3 shows Pc at 761 Hz, P2 is 4.42–6.07 s, 2.5%")]
    assert nums == [761.0, 4.42, -6.07, 2.5] or nums == [761.0, 4.42, 6.07, 2.5]


def test_verifier_flags_invented_numbers():
    _, s = session_for(["oscillation"], seed=21)
    ev = s.run("detect_oscillation")
    f_ok = ev.result["events"][0]["freq_hz"]
    good = Finding("osc", f"Pc oscillates at {f_ok:.0f} Hz", "combustion_oscillation", "critical", "Pc",
                   evidence=[ev.id])
    bad = Finding("osc", f"Pc oscillates at {f_ok + 137:.0f} Hz", "combustion_oscillation", "critical", "Pc",
                  evidence=[ev.id])
    none = Finding("osc", "Pc oscillates", "combustion_oscillation", "critical", "Pc", evidence=["E99"])
    summary = verify_findings([good, bad, none], s)
    assert good.verification["status"] == "verified"
    assert bad.verification["status"] == "partial"
    assert bad.verification["ungrounded_numbers"] == [f"{f_ok + 137:.0f}"]
    assert none.verification["status"] == "unsupported"
    assert summary["verified"] == 1


def test_unit_scaling_is_accepted():
    _, s = session_for(["valve_delay"], seed=4)
    ev = s.run("measure_valve_response")
    fu = next(e for e in ev.result["events"] if e["command"] == "cmd_fu" and e["edge"] == "open")
    f = Finding("late", f"response after {fu['latency_ms'] / 1000:.4f} s", "valve_response", "warning",
                "P_fu_inj", evidence=[ev.id])
    verify_findings([f], s)
    assert f.verification["status"] == "verified"


def test_rule_agent_all_claims_verified_and_detects_everything():
    run, s = session_for(list(ANOMALY_TYPES), seed=3)
    res = RuleAgent("en").run(s)
    assert res.verification["verified"] == res.verification["n_findings"]
    m = match(res.findings, run.truth)
    assert all(t["detected"] for t in m["truth"]), m
    assert not m["false_positives"], m["false_positives"]


def test_report_and_reproduce(tmp_path):
    run = generate_run(3, ["oscillation", "overtemp"])
    paths = run.save(tmp_path)
    s = Session.open(paths["run"])
    res = RuleAgent().run(s)
    out = write_report(s, res, tmp_path / "report.html")
    html = out["html"].read_text()
    assert "E1" in html and "data:image/png;base64" in html
    r = reproduce(out["json"])
    assert r["data_hash_matches"]
    assert r["n_reproduced"] == r["n"] == len(s.ledger)


def test_tdms_roundtrip(tmp_path):
    pytest.importorskip("nptdms")
    run = generate_run(2, ["overtemp"])
    paths = run.save(tmp_path, fmt="tdms")
    s = Session.open(paths["run"])
    assert set(run.data.columns) - {"time"} <= set(s.channels)
    assert s.run("check_redlines").result["n_violations"] == 1


def test_llm_agent_loop_with_verifier_feedback():
    """A scripted 'LLM' first submits an invented number, gets rejected, then fixes it."""
    run, s = session_for(["oscillation"], seed=21)

    def call(name, args, i):
        return {"content": "", "tool_calls": [{"id": f"c{i}", "name": name, "arguments": args}]}

    def last_evidence(messages):
        for m in reversed(messages):
            if m["role"] == "tool" and "evidence_id" in m["content"]:
                return json.loads(m["content"])
        raise AssertionError

    def submit(freq_fn):
        def reply(messages):
            ev = last_evidence(messages) if freq_fn else None
            e = ev["result"]["events"][0] if ev else None
            freq = freq_fn(e) if e else 0
            return call("submit_report", {
                "summary": "one oscillation",
                "findings": [{"title": "oscillation", "statement": f"Pc oscillates at {freq:.0f} Hz",
                              "category": "combustion_oscillation", "severity": "critical", "channel": "Pc",
                              "t_start": e["t_start"], "t_end": e["t_end"], "evidence": [ev["evidence_id"]]}],
            }, 9)
        return reply

    def resubmit(messages):
        ev = next(json.loads(m["content"]) for m in reversed(messages)
                  if m["role"] == "tool" and m["name"] == "detect_oscillation")
        e = ev["result"]["events"][0]
        return call("submit_report", {
            "summary": "one oscillation",
            "findings": [{"title": "oscillation", "statement": f"Pc oscillates at {e['freq_hz']:.0f} Hz",
                          "category": "combustion_oscillation", "severity": "critical", "channel": "Pc",
                          "t_start": e["t_start"], "t_end": e["t_end"], "evidence": [ev["evidence_id"]]}],
        }, 10)

    backend = ScriptedBackend([
        call("check_sensor_health", {}, 1),
        call("detect_oscillation", {"channel": "Pc"}, 2),
        submit(lambda e: e["freq_hz"] + 250),  # hallucinated frequency
        resubmit,
    ])
    res = LLMAgent(backend, lang="en").run(s)
    assert backend.calls == 4
    assert res.verification["verified"] == 1
    rejection = [t for t in res.transcript if t["role"] == "tool" and "Verifier rejected" in t["content"]]
    assert rejection


def test_benchmark_rule_agent():
    res = run_benchmark(lambda: RuleAgent("en"), n=12, seed=100)
    s = res["summary"]
    assert s["recall"] >= 0.9
    assert s["precision"] >= 0.9


def test_openai_backend_wire_format(monkeypatch):
    """OpenAI-compatible backend: request shape, no forced temperature, usage + first-draft tracking."""
    import httpx

    from groundline.agent import make_agent

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/chat/completions")
        assert "temperature" not in json
        tool_msgs = [m for m in json["messages"] if m["role"] == "tool"]
        if not tool_msgs:
            tc = [{"id": "a1", "type": "function", "function": {"name": "check_redlines", "arguments": "{}"}}]
        else:
            ev = [__import__("json").loads(m["content"]) for m in tool_msgs if "evidence_id" in m["content"]][-1]
            n_sub = sum(1 for m in json["messages"] if m["role"] == "assistant"
                        and any(t["function"]["name"] == "submit_report" for t in m.get("tool_calls", [])))
            num = "123.45" if n_sub == 0 else str(ev["result"]["persistence_s"])
            args = {"summary": "s", "findings": [{"title": "t", "statement": f"persistence {num} s",
                                                  "category": "observation", "severity": "info",
                                                  "evidence": [ev["evidence_id"]]}]}
            tc = [{"id": f"s{n_sub}", "type": "function",
                   "function": {"name": "submit_report", "arguments": __import__("json").dumps(args)}}]
        body = {"choices": [{"message": {"content": "", "tool_calls": tc}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    _, s = session_for(["overtemp"], seed=2)
    res = make_agent("openai", "en", "fake", "http://x/v1", "k").run(s)
    assert res.verification["verified"] == 1
    assert res.agent["first_submission"]["numbers_grounded"] == 0
    assert res.agent["fix_rounds_used"] == 1
    assert res.agent["usage"]["requests"] == 3


def test_dotenv_parsing_and_precedence(tmp_path, monkeypatch):
    from groundline.config import load_dotenv, parse_dotenv

    env = parse_dotenv('# c\nexport A=1\nB="two words"\nC=3 # note\nD=\'x\'\nbad line\n')
    assert env == {"A": "1", "B": "two words", "C": "3", "D": "x"}
    (tmp_path / ".env").write_text("GROUNDLINE_LLM_MODEL=from-file\nGROUNDLINE_LANG=en\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    monkeypatch.setenv("GROUNDLINE_LANG", "zh")  # shell wins over file
    monkeypatch.delenv("GROUNDLINE_LLM_MODEL", raising=False)
    assert load_dotenv() == tmp_path / ".env"
    import os
    assert os.environ["GROUNDLINE_LLM_MODEL"] == "from-file"
    assert os.environ["GROUNDLINE_LANG"] == "zh"


def test_openai_responses_backend(monkeypatch):
    """Responses API: instructions + incremental input with previous_response_id, function tools, reasoning."""
    import json as J

    import httpx

    from groundline.agent import OpenAIResponses, make_agent

    monkeypatch.delenv("GROUNDLINE_OPENAI_API", raising=False)
    monkeypatch.delenv("GROUNDLINE_LLM_BASE_URL", raising=False)
    seen = []
    state = {"n": 0, "submits": 0, "last_ev": None}

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/responses")
        assert json["instructions"] and json["tools"][0]["type"] == "function"
        assert json["reasoning"] == {"effort": "low"}
        seen.append(json)
        state["n"] += 1
        if state["n"] == 1:
            assert "previous_response_id" not in json
            assert json["input"][0]["role"] == "user"
            out = [{"type": "reasoning", "summary": []},
                   {"type": "function_call", "call_id": "c1", "name": "check_redlines", "arguments": "{}"}]
        else:
            assert json["previous_response_id"] == f"resp_{state['n'] - 1}"
            # only new items are sent: tool outputs, never the assistant turn again
            assert all(i.get("type") == "function_call_output" for i in json["input"]), json["input"]
            for i in json["input"]:
                if "evidence_id" in i["output"]:
                    state["last_ev"] = J.loads(i["output"])
            ev = state["last_ev"]
            num = "123.45" if state["submits"] == 0 else str(ev["result"]["persistence_s"])
            state["submits"] += 1
            args = {"summary": "s", "findings": [{"title": "t", "statement": f"persistence {num} s",
                                                  "category": "observation", "severity": "info",
                                                  "evidence": [ev["evidence_id"]]}]}
            out = [{"type": "function_call", "call_id": f"s{state['n']}", "name": "submit_report",
                    "arguments": J.dumps(args)}]
        body = {"id": f"resp_{state['n']}", "output": out, "usage": {"input_tokens": 100, "output_tokens": 20}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("GROUNDLINE_LLM_REASONING_EFFORT", "low")
    agent = make_agent("openai", "en", "gpt-5.6-sol", None, "k")
    assert isinstance(agent.backend, OpenAIResponses)
    _, s = session_for(["overtemp"], seed=2)
    res = agent.run(s)
    assert res.verification["verified"] == 1
    assert res.agent["first_submission"]["numbers_grounded"] == 0
    assert res.agent["usage"] == {"input_tokens": 300, "output_tokens": 60, "requests": 3}
    # third request carries the verifier rejection as a function_call_output
    assert "Verifier rejected" in seen[2]["input"][0]["output"]


# ---------------------------------------------------------------------------- real data: HANARO solid motor
HANARO = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "hanaro_knsb" / "run.csv"


def _toy_session(cols, fs=100.0, meta_channels=None):
    import pandas as pd

    df = pd.DataFrame(cols)
    meta = {"sample_rate_hz": fs, "channels": meta_channels or {}}
    return Session(df, meta, None, {})


def test_slow_channel_is_not_searched_above_its_nyquist():
    t = np.arange(0, 10, 0.01)
    pc = np.where((t > 3) & (t < 7), 40.0, 1.0)
    s = _toy_session({"time": t, "Pc": pc},
                     meta_channels={"Pc": {"unit": "bar", "kind": "pressure", "native_rate_hz": 10}})
    r = s.run("detect_oscillation", channel="Pc", fmin=5).result
    assert r["applicable"] is False and not r["events"]


def test_recurring_gaps_are_reported_once_and_quiet_flatlines_ignored():
    t = np.arange(0, 20, 0.01)
    f = np.where((t > 8) & (t < 12), 1000.0, 0.0) + np.random.default_rng(0).normal(0, 2, t.size)
    for k in range(10):  # a dropped DAQ frame every 2 s
        f[(t >= 2 * k + 0.5) & (t < 2 * k + 0.6)] = np.nan
    pc = np.round(np.where((t > 8) & (t < 12), 40.0, 1.29), 3)  # quantized, perfectly quiet before firing
    pc[(t > 8) & (t < 12)] += np.sin(t[(t > 8) & (t < 12)])  # moving while firing
    s = _toy_session({"time": t, "Pc": pc, "F_thrust": f},
                     meta_channels={"Pc": {"unit": "bar", "kind": "pressure"},
                                    "F_thrust": {"unit": "N", "kind": "force"}})
    issues = s.run("check_sensor_health").result["issues"]
    kinds = [(i["channel"], i["kind"]) for i in issues]
    assert kinds.count(("F_thrust", "recurring_nan_gaps")) == 1
    assert not any(k == "nan_gap" for _, k in kinds)
    assert ("Pc", "flatline") not in kinds  # steady ambient reading before ignition is not a fault


def test_steep_edge_on_slow_logger_is_not_a_spike():
    t = np.arange(0, 10, 0.01)
    native = np.arange(0, 10, 0.1)
    p_native = np.interp(native, [0, 4, 4.5, 8, 8.5, 10], [1, 1, 45, 40, 1, 1])
    pc = np.interp(t, native, p_native)
    s = _toy_session({"time": t, "Pc": pc},
                     meta_channels={"Pc": {"unit": "bar", "kind": "pressure", "native_rate_hz": 10}})
    assert not s.run("check_sensor_health").result["issues"]


def test_pulse_metrics_impulse():
    t = np.arange(0, 10, 0.01)
    f = np.where((t >= 3) & (t < 7), 500.0, 0.0) + 5.0  # 4 s at 500 N on a 5 N baseline
    s = _toy_session({"time": t, "F_thrust": f}, meta_channels={"F_thrust": {"unit": "N", "kind": "force"}})
    r = s.run("pulse_metrics").result
    assert r["baseline"] == pytest.approx(5.0)
    assert r["peak"] == pytest.approx(500.0)
    assert r["integral"] == pytest.approx(2000.0, rel=0.01)


@pytest.mark.skipif(not HANARO.exists(), reason="example data not present")
def test_hanaro_static_fire_matches_team_processing():
    s = Session.open(HANARO)
    res = RuleAgent().run(s)
    assert res.verification["verified"] == len(res.findings)
    pm = next(e for e in s.ledger if e.tool == "pulse_metrics").result
    # HANARO's own processed output: peak 2221.7 N, 6411 N·s over its 4.35 s window
    assert pm["peak"] == pytest.approx(2221.7, rel=0.005)
    assert pm["integral"] == pytest.approx(6411, rel=0.02)
    kinds = {(f.channel, f.category) for f in res.findings}
    assert ("F_thrust", "sensor_fault") in kinds  # recurring DAQ dropouts and the pre-test glitch
    assert not any(f.channel == "Pc" and f.category == "sensor_fault" for f in res.findings)


@pytest.mark.skipif(not HANARO.exists(), reason="example data not present")
def test_ignition_ramp_is_not_an_oscillation():
    s = Session.open(HANARO)
    r = s.run("detect_oscillation", channel="F_thrust").result
    assert not r["events"]


# ---------------------------------------------------------------------------- leaderboard / weak models
def test_invalid_submit_is_sent_back_not_accepted():
    """A model that sends broken JSON must be told so, not have an empty report accepted."""
    _, s = session_for([], seed=4)
    good = {"summary": "nominal", "findings": [{"title": "sequence", "statement": "nominal run",
                                                "category": "observation", "severity": "info",
                                                "evidence": ["E2"]}]}
    backend = ScriptedBackend([
        {"content": "", "tool_calls": [{"id": "a", "name": "submit_report",
                                        "arguments": {"__invalid_json__": "{\"summary\": \"nomi"}}]},
        {"content": "", "tool_calls": [{"id": "b", "name": "submit_report",
                                        "arguments": {"summary": "x", "findings": "not json"}}]},
        {"content": "", "tool_calls": [{"id": "c", "name": "submit_report",
                                        "arguments": {**good, "findings": json.dumps(good["findings"])}}]},
    ])
    res = LLMAgent(backend, lang="en").run(s)
    assert backend.calls == 3
    assert res.agent["submitted"] and len(res.findings) == 1
    errors = [t for t in res.transcript if t["role"] == "tool" and "error" in t["content"]]
    assert len(errors) == 2


def test_agent_that_never_submits_counts_as_missed(tmp_path):
    from groundline.evaluate import summarize

    def silent():
        return LLMAgent(ScriptedBackend([]), lang="en", max_steps=2)

    res = run_benchmark(silent, n=2, seed=1001, keep_going=True)
    s = summarize(res["runs"])
    assert s["reports_submitted"] == 0
    assert s["recall_all_runs"] == 0


def test_leaderboard_runs_and_imports(tmp_path):
    from groundline.evaluate import save
    from groundline.leaderboard import run_leaderboard

    prev = run_benchmark(lambda: RuleAgent("en"), n=2, seed=1000)
    save(prev, tmp_path / "old.json")
    cfg = {"n": 2, "seed": 1000, "lang": "en", "models": [
        {"name": "rule", "agent": "rule"},
        {"name": "imported", "agent": "rule", "from": str(tmp_path / "old.json")},
    ]}
    md = run_leaderboard(cfg, tmp_path / "lb")
    table = md.read_text()
    assert "| rule | 2/2 | 100% / 100% |" in table and "| imported | 2/2 | 100% / 100% |" in table
    # second call reuses results
    assert run_leaderboard(cfg, tmp_path / "lb").read_text() == table


def test_leaderboard_entry_ignores_env_reasoning_effort(monkeypatch):
    from groundline.leaderboard import make_entry_agent

    monkeypatch.setenv("GROUNDLINE_LLM_REASONING_EFFORT", "high")
    a = make_entry_agent({"name": "local", "agent": "openai", "model": "qwen2.5:7b-16k",
                          "base_url": "http://localhost:11434/v1", "api_key": "ollama"}, "zh")
    assert a.backend.reasoning_effort is None


def test_loose_match_counts_channelless_claim_with_right_category():
    run, _ = session_for(["overtemp"], seed=5)
    f = Finding("coolant outlet over redline", "…", "redline_violation", "critical", None, None, None, ["E1"])
    m = match([f], run.truth)
    assert not m["truth"][0]["detected"] and m["truth"][0]["detected_loose"]


def test_oscillation_window_longer_than_span_is_clamped():
    _, s = session_for(["oscillation"], seed=21)
    r = s.run("detect_oscillation", channel="Pc", window_s=400).result
    assert r["window_s"] <= r["t_end"] - r["t_start"] + 1e-9


def test_hyphenated_identifiers_are_not_numbers():
    vals = [v for _, v, _ in extract_numbers("Tested SYN-1013 and run_42: peak 5.2 bar, -3 dB")]
    assert vals == [5.2, -3.0]


# ---------------------------------------------------------------------------- semantic checks
def _pc_session_with_pulse():
    t = np.arange(0, 10, 0.01)
    f = np.where((t >= 3) & (t < 7), 500.0, 0.0) + 5.0
    s = _toy_session({"time": t, "F_thrust": f}, meta_channels={"F_thrust": {"unit": "N", "kind": "force"}})
    return s, s.run("pulse_metrics")


def _check(s, stmt, ev):
    from groundline.findings import verify_finding

    return verify_finding(Finding("t", stmt, "observation", "info", "F_thrust", None, None, [ev.id]), s)


def test_semantic_accepts_correct_reading():
    s, ev = _pc_session_with_pulse()
    r = ev.result
    v = _check(s, f"峰值 {r['peak']:.1f} N，工作时间 {r['action_time_s']:.2f} s，总冲 {r['integral']:.0f} N·s，"
                  f"{r['t_start']:.2f}–{r['t_end']:.2f} s，按峰值的 {r['start_pct']:g}% 截取。", ev)
    assert v["status"] == "verified", v


def test_semantic_flags_real_value_in_wrong_role_or_unit():
    s, ev = _pc_session_with_pulse()
    r = ev.result
    # the action time written as the peak: a real number, wrong meaning
    v = _check(s, f"peak of {r['action_time_s']:.2f}", ev)
    assert v["status"] == "partial" and v["semantic_problems"]
    # a time written in Hz
    v = _check(s, f"at {r['t_peak']:.2f} Hz", ev)
    assert v["mismatched_numbers"]
    # the peak force written with a time unit
    v = _check(s, f"{r['peak']:.1f} s", ev)
    assert v["mismatched_numbers"]


def test_role_word_must_lead_into_the_number():
    from groundline.semantics import role_before

    t = "T_cool_out 持续超出红线 700 K"
    assert role_before(t, t.index("700")) is None
    t = "峰值为 742.3 K"
    assert role_before(t, t.index("742"))[0] == "peak"


def test_verifier_bench_catches_more_than_grounding_without_false_alarms():
    from groundline.verifier_bench import run_verifier_bench

    r = run_verifier_bench(n=3, seed=1000, langs=("zh",))
    assert r["clean"]["flagged_full"] == 0
    for k in ("fabricated", "swapped", "wrong_unit"):
        st = r["mutations"][k]
        assert st["caught_full"] >= st["caught_grounding"]
    assert r["mutations"]["wrong_unit"]["caught_full"] > 0 and r["mutations"]["swapped"]["caught_full"] > 0


def test_llm_runs_can_be_reverified_later():
    from groundline.evaluate import reverify

    def agent():
        return LLMAgent(ScriptedBackend([
            {"content": "", "tool_calls": [{"id": "a", "name": "submit_report", "arguments": {
                "summary": "x", "findings": [{"title": "seq", "statement": "ignition 999.5 s", "category": "observation",
                                              "severity": "info", "evidence": ["E2"]}]}}]},
        ]), lang="en", fix_rounds=0)

    res = run_benchmark(agent, n=1, seed=1000)
    assert "ledger" in res["runs"][0]
    before = res["runs"][0]["verification"]
    after = reverify(json.loads(json.dumps(res)))["runs"][0]["verification"]
    assert before["numbers_grounded"] == after["numbers_grounded"] == 0


def test_leaderboard_resumes_after_model_server_goes_away(tmp_path, monkeypatch):
    import groundline.leaderboard as lb

    class ConnectError(Exception):  # same name as httpx's, which is what matters
        pass

    calls = {"n": 0, "fail_from": 2}

    class Flaky:
        def run(self, s):
            calls["n"] += 1
            if calls["n"] >= calls["fail_from"]:
                raise ConnectError("[Errno 61] Connection refused")
            return RuleAgent("en").run(s)

    monkeypatch.setattr(lb, "make_entry_agent", lambda e, lang: Flaky())
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = {"n": 3, "seed": 1000, "lang": "en", "models": [{"name": "local", "agent": "openai"}]}
    lb.run_leaderboard(cfg, tmp_path)
    part = json.loads((tmp_path / "local.json").read_text())
    assert part["incomplete"] and len(part["runs"]) == 1
    calls.update(n=0, fail_from=99)  # server is back
    md = lb.run_leaderboard(cfg, tmp_path).read_text()
    done = json.loads((tmp_path / "local.json").read_text())
    assert not done.get("incomplete") and len(done["runs"]) == 3 and calls["n"] == 2
    assert "| local | 3/3 |" in md


def test_leaderboard_checkpoints_every_run_and_survives_ctrl_c(tmp_path, monkeypatch):
    import groundline.leaderboard as lb

    calls = {"n": 0}

    class Slow:
        def run(self, s):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            return RuleAgent("en").run(s)

    monkeypatch.setattr(lb, "make_entry_agent", lambda e, lang: Slow())
    cfg = {"n": 4, "seed": 1000, "lang": "en", "models": [{"name": "local", "agent": "openai"}]}
    with pytest.raises(SystemExit):
        lb.run_leaderboard(cfg, tmp_path)
    part = json.loads((tmp_path / "local.json").read_text())
    assert part["incomplete"] and len(part["runs"]) == 2


def test_run_time_budget_and_reply_cap():
    from groundline.agent import OpenAICompatible
    from groundline.leaderboard import make_entry_agent

    _, s = session_for([], seed=4)
    stall = [{"content": "thinking...", "tool_calls": []}] * 50
    res = LLMAgent(ScriptedBackend(stall), lang="en", max_seconds=0.0).run(s)
    assert res.agent["timed_out"] and not res.agent["submitted"]
    a = make_entry_agent({"name": "x", "agent": "openai", "model": "m", "base_url": "http://localhost:11434/v1"}, "en")
    assert isinstance(a.backend, OpenAICompatible) and a.backend.max_tokens == 2048 and a.max_seconds == 1200


def test_compound_units_are_read_whole_and_cancelled():
    from groundline.semantics import _norm_unit, unit_after

    assert unit_after("39.6 MPa·s。", 4) == "MPa·s"
    assert unit_after("50 kg/s·s 和", 2) == "kg/s·s"
    assert _norm_unit("kg/s·s") == _norm_unit("kg")  # a flow integral is a mass
    assert _norm_unit("kg·s") != _norm_unit("kg/s·s")


# ---------------------------------------------------------------------------- MCP server / registry metadata
def test_registry_metadata_is_consistent():
    import pathlib
    import re as _re

    root = pathlib.Path(__file__).resolve().parents[1]
    server = json.loads((root / "server.json").read_text())
    version = _re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), _re.M).group(1)
    assert server["version"] == version and server["packages"][0]["version"] == version
    assert server["packages"][0]["identifier"] == "groundline"
    assert f"mcp-name: {server['name']}" in (root / "README.md").read_text()
    assert len(server["description"]) <= 100


def test_mcp_server_round_trip(tmp_path):
    import asyncio

    pytest.importorskip("mcp")
    from groundline.mcp_server import build_server

    run = generate_run(21, ["oscillation"])
    paths = run.save(tmp_path / "run")
    srv = build_server()

    async def go():
        def payload(res):
            blocks = res[0] if isinstance(res, tuple) else res
            return json.loads(blocks[0].text)

        opened = payload(await srv.call_tool("open_run", {"run_path": str(paths["run"])}))
        sid = opened["run_id"]
        ev = payload(await srv.call_tool("run_analysis", {"run_id": sid, "tool": "detect_oscillation",
                                                          "params": {"channel": "Pc"}}))
        f = ev["result"]["events"][0]
        good = {"title": "osc", "statement": f"Pc oscillates at {f['freq_hz']:.0f} Hz", "category": "combustion_oscillation",
                "severity": "critical", "channel": "Pc", "evidence": [ev["evidence_id"]]}
        bad = {**good, "statement": f"Pc oscillates at {f['freq_hz'] + 300:.0f} Hz"}
        v = payload(await srv.call_tool("verify", {"run_id": sid, "findings": [good, bad]}))
        # a bridge that strips the id: with one open run the server still knows which one is meant
        v2 = payload(await srv.call_tool("verify", {"findings": [good, bad]}))
        assert v2["summary"] == v["summary"]
        return v

    v = asyncio.run(go())
    assert v["summary"]["verified"] == 1 and v["summary"]["partial"] == 1
