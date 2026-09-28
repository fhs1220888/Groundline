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
