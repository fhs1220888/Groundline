/* Groundline verifier, JavaScript port for the web demo.
 *
 * A line-by-line port of the number checks in src/groundline/findings.py (grounding) and
 * src/groundline/semantics.py (units and role words). docs/demo/parity_check.mjs runs it against
 * verdicts produced by the Python verifier; keep the two in step when either changes.
 */
(function (root) {
  "use strict";

  const NUM_RE = /(?<![A-Za-z_\d.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?/g;
  const SCALES = [1.0, 1000.0, 0.001, 100.0, 0.01, 60.0];
  const isAlpha = (ch) => /\p{L}/u.test(ch || "");

  function numberMatches(text) {
    const out = [];
    text = text || "";
    NUM_RE.lastIndex = 0;
    let m;
    while ((m = NUM_RE.exec(text)) !== null) {
      const i = m.index;
      if (i >= 2 && "-_".includes(text[i - 1]) && isAlpha(text[i - 2])) continue;
      out.push({ text: m[0], start: i, end: i + m[0].length });
    }
    return out;
  }

  function matches(value, decimals, v) {
    const a = Math.abs(value);
    for (const sc of SCALES) {
      const x = Math.abs(v * sc);
      if (Math.abs(a - x) <= Math.max(0.5 * Math.pow(10, -decimals), 0.01 * x, 1e-9)) return sc;
    }
    return null;
  }

  // ------------------------------------------------------------------ evidence fields
  const TIME_KEYS = new Set(["t", "time", "t_start", "t_end", "t_peak", "t_cmd", "t_response", "t_center", "t_s"]);

  function kindOf(key, unit) {
    const k = key.toLowerCase();
    if (k.startsWith("#len")) return "count";
    if (k.endsWith("_hz") || ["hz", "freq", "frequency"].includes(k)) return "freq";
    if (k.endsWith("_pct") || k.endsWith("pct") || k === "percent") return "percent";
    if (TIME_KEYS.has(k) || k.startsWith("t_") || k.endsWith("_s") || k.endsWith("_ms")) return "time";
    if (k.startsWith("n_") || k.includes("count") || k.endsWith("samples") || ["windows", "spikes", "gaps"].includes(k))
      return "count";
    if (unit) return "physical";
    return "plain";
  }

  function evidenceFields(obj, path = "", key = "", unit = null) {
    const out = [];
    if (obj === null || obj === undefined || typeof obj === "boolean") return out;
    if (typeof obj === "number") {
      out.push({ value: obj, key, path, unit, kind: kindOf(key, unit) });
    } else if (Array.isArray(obj)) {
      out.push({ value: obj.length, key: `#len(${key})`, path: `len(${path})`, unit: null, kind: "count" });
      obj.forEach((v, i) => out.push(...evidenceFields(v, `${path}[${i}]`, key, unit)));
    } else if (typeof obj === "object") {
      const base = typeof obj.unit === "string" ? obj.unit : null;
      for (const [k, v] of Object.entries(obj)) {
        const u = typeof obj[`${k}_unit`] === "string" ? obj[`${k}_unit`] : base;
        out.push(...evidenceFields(v, path ? `${path}.${k}` : k, k, u));
      }
    }
    return out;
  }

  // ------------------------------------------------------------------ reading the text
  const BASE_U = "(?:MPa|kPa|Pa|bar|psi|degC|°C|℃|kg|Hz|hz|ms|sec|s|K|N|g|m|W|J|V|A|rpm)";
  const UNIT_SRC =
    "\\s*(毫秒|秒|赫兹|%|％|个采样点|个|次|段|条|处|samples?|windows?|spikes?|gaps?" +
    "|" + BASE_U + "(?:\\s?[·*/]\\s?" + BASE_U + ")*)(?![A-Za-z])";
  const TIME_U = new Set(["毫秒", "秒", "ms", "sec", "s"]);
  const FREQ_U = new Set(["赫兹", "Hz", "hz"]);
  const PCT_U = new Set(["%", "％"]);
  const COUNT_U = new Set(["个采样点", "个", "次", "段", "条", "处", "sample", "samples", "window", "windows",
    "spike", "spikes", "gap", "gaps"]);

  function unitMatchAt(text, pos) {
    const re = new RegExp(UNIT_SRC, "y");
    re.lastIndex = pos;
    const m = re.exec(text);
    return m ? { unit: m[1], end: re.lastIndex } : null;
  }
  const unitAfter = (text, end) => { const m = unitMatchAt(text, end); return m ? m.unit : null; };

  function unitKind(u) {
    if (u === null) return null;
    if (TIME_U.has(u)) return "time";
    if (FREQ_U.has(u)) return "freq";
    if (PCT_U.has(u)) return "percent";
    if (COUNT_U.has(u)) return "count";
    return "physical";
  }

  function dims(u) {
    u = u.trim();
    u = { "°C": "degC", "℃": "degC" }[u] || u;
    const powers = {};
    let sign = 1;
    for (const tok of u.match(/[·*/]|[^·*/\s]+/g) || []) {
      if (tok === "·" || tok === "*") sign = 1;
      else if (tok === "/") sign = -1;
      else { powers[tok] = (powers[tok] || 0) + sign; sign = 1; }
    }
    return Object.keys(powers).filter((k) => powers[k]).sort().map((k) => `${k}^${powers[k]}`).join(" ");
  }
  const normUnit = (u) => (u === null || u === undefined ? null : dims(u));

  const ROLES = {
    peak: [["峰值", "最大值", "最大", "最高", "peak", "maximum", "max"], /peak|max/],
    duration: [["持续时间", "持续", "时长", "历时", "工作时间", "lasting", "duration", "lasted"],
      /duration|action_time|total_s|longest|elapsed|dur/],
    mean: [["平均值", "平均", "均值", "mean", "average", "averages"], /mean|average|level|median/],
    integral: [["总冲量", "总冲", "冲量", "积分", "impulse", "integral", "integrates to"], /integral|impulse/],
    freq: [["频率", "frequency"], /freq|hz/],
    deviation: [["偏差", "偏离", "deviation", "deviates"], /dev/],
    latency: [["延迟", "滞后", "latency", "delay"], /latenc|lat_|delay/],
  };
  const GAP_FILLER = /\s+|[:：=≈~]|达到|为|约|了|达|近|是|值|\b(?:of|about|approximately|approx\.?|around|is|was|at|reached|reaching|reaches|to|by|a|an|the)\b/gi;
  const CLAUSE_END = new Set("，,;；。\n（(）)".split(""));
  const RANGE_SEP = /^\s*(?:[–—~\-]|至|到|to)\s*[-+]?\d/;
  const isAsciiAlphaWord = (w) => /^[A-Za-z]+$/.test(w);

  function roleBefore(text, start) {
    let lo = start;
    while (lo > 0 && !CLAUSE_END.has(text[lo - 1]) && start - lo < 40) lo -= 1;
    const clause = text.slice(lo, start);
    const lower = clause.toLowerCase();
    let best = null;
    for (const [role, [words]] of Object.entries(ROLES)) {
      for (const w of words) {
        const i = lower.lastIndexOf(w.toLowerCase());
        if (i < 0) continue;
        if (isAsciiAlphaWord(w)) {
          const a = i - 1, b = i + w.length;
          if ((a >= 0 && isAlpha(clause[a])) || (b < clause.length && isAlpha(clause[b]))) continue;
        }
        const end = i + w.length;
        if (best === null || end > best.end) best = { end, role, word: w };
      }
    }
    if (best === null) return null;
    if (clause.slice(best.end).replace(GAP_FILLER, "")) return null;
    return { role: best.role, word: best.word };
  }

  function isRangeEndpoint(text, start, end) {
    let after = text.slice(end);
    const u = unitMatchAt(after, 0);
    if (u) after = after.slice(u.end);
    if (RANGE_SEP.test(after)) return true;
    const before = text.slice(0, start).replace(/\s+$/, "");
    return /(?:[–—~]|至|到|\bto|\d\s*-)$/.test(before);
  }

  function unitOk(tk, tu, f, scale) {
    if (tk === null) return true;
    if (tk === "time") return f.kind === "time";
    if (tk === "freq") return f.kind === "freq";
    if (tk === "percent") return f.kind === "percent" || (scale === 100.0 && f.kind === "plain");
    if (tk === "count") return f.kind === "count" || f.kind === "plain";
    if (f.kind === "plain") return true;
    if (f.kind !== "physical") return false;
    return normUnit(f.unit) === normUnit(tu);
  }

  function checkNumber(text, start, end, candidates) {
    const tu = unitAfter(text, end);
    const tk = unitKind(tu);
    const rb = roleBefore(text, start);
    const role = rb === null || isRangeEndpoint(text, start, end) ? null : rb;
    const byUnit = candidates.filter(([f, sc]) => unitOk(tk, tu, f, sc));
    const tok = text.slice(start, end);
    if (candidates.length && !byUnit.length) {
      const f = candidates[0][0];
      const what = f.kind !== "physical" ? f.kind : `unit ${f.unit}`;
      return { ok: false, unit: tu, role: rb && rb.role, field: f.path,
               problem: `'${tok} ${tu}' matches only ${f.path} (${what})` };
    }
    let pool = byUnit.length ? byUnit : candidates;
    if (role && pool.length) {
      const rx = ROLES[role.role][1];
      const fitting = pool.filter(([f]) => rx.test(f.key.toLowerCase()));
      if (!fitting.length) {
        const f = pool[0][0];
        return { ok: false, unit: tu, role: role.role, field: f.path,
                 problem: `'${role.word} ${tok}' is read as ${role.role}, but the value comes from ${f.path}` };
      }
      pool = fitting;
    }
    return { ok: true, unit: tu, role: role && role.role, field: pool.length ? pool[0][0].path : null, problem: null };
  }

  // ------------------------------------------------------------------ one finding
  /* ledger: {E1: {result, params}, ...}; finding: {title, statement, evidence: ["E2", ...]} */
  function verifyFinding(finding, ledger) {
    const problems = [];
    const cited = [];
    const missing = [];
    for (const id of finding.evidence || []) (ledger[id] ? cited.push([id, ledger[id]]) : missing.push(id));
    if (missing.length) problems.push(`unknown evidence id(s): ${missing.join(", ")}`);
    const fields = [];
    for (const [id, ev] of cited) {
      fields.push(...evidenceFields(ev.result, id));
      fields.push(...evidenceFields(ev.params || {}, `${id}.params`));
    }
    const text = `${finding.title || ""}\n${finding.statement || ""}`;
    const numbers = [];
    for (const m of numberMatches(text)) {
      const norm = m.text.replace("−", "-");
      const val = parseFloat(norm);
      if (Number.isNaN(val)) continue;
      const mant = norm.toLowerCase().split("e")[0];
      const dec = mant.includes(".") ? mant.split(".")[1].length : 0;
      const cands = [];
      for (const fl of fields) { const sc = matches(val, dec, fl.value); if (sc !== null) cands.push([fl, sc]); }
      // closest value first, so a message names the field the writer most likely meant (stable, as in Python)
      cands.sort((a, b) => Math.abs(Math.abs(val) - Math.abs(a[0].value * a[1])) - Math.abs(Math.abs(val) - Math.abs(b[0].value * b[1])));
      const sem = cands.length ? checkNumber(text, m.start, m.end, cands) : null;
      let best = null;
      if (cands.length) best = cands.find(([fl]) => sem && fl.path === sem.field) || cands[0];
      numbers.push({
        text: m.text, value: val, start: m.start, end: m.end,
        inTitle: m.start < (finding.title || "").length,
        grounded: cands.length > 0,
        consistent: cands.length > 0 && sem.ok,
        matched: best && { value: best[0].value, scale: best[1], field: best[0].path },
        unit: sem ? sem.unit : null, role: sem ? sem.role : null,
        semanticProblem: sem ? sem.problem : null,
      });
    }
    const ungrounded = numbers.filter((n) => !n.grounded).map((n) => n.text);
    const mismatched = numbers.filter((n) => n.grounded && !n.consistent).map((n) => n.text);
    const status = !cited.length ? "unsupported" : ungrounded.length || mismatched.length || problems.length ? "partial" : "verified";
    return { status, numbers, ungrounded, mismatched, problems, titleLength: (finding.title || "").length + 1 };
  }

  const api = { verifyFinding, evidenceFields, numberMatches };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.GroundlineVerifier = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
