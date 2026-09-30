"""Build the web demo (docs/demo/index.html) from template.html.

The page is static: the verifier runs in the browser (verifier.js, a port of the Python checks that
parity_check.mjs holds to the Python verdicts), on evidence computed here by the real tools from the
HANARO static-fire data. Run from the repository root:

    python docs/demo/build_demo.py            # writes docs/demo/index.html and docs/demo/report_hanaro.html
    python docs/demo/build_demo.py --fragment out.html   # body-only copy for hosts that add their own <head>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from groundline.agent import RuleAgent
from groundline.report import write_report
from groundline.session import Session

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def ledger() -> tuple[dict, Session]:
    s = Session.open(ROOT / "examples" / "hanaro_knsb" / "run.csv")
    s.run("describe_data")
    s.run("segment_phases")
    s.run("check_sensor_health")
    s.run("pulse_metrics", channel="F_thrust")
    pm = s.ledger[-1].result
    s.run("channel_stats", channel="Pc", t_start=pm["t_start"], t_end=pm["t_end"])
    out = {}
    for e in s.ledger:
        out[e.id] = {"tool": e.tool, "params": e.params, "result": e.result,
                     "figure": f"data:image/png;base64,{e.figure_png}" if e.figure_png and e.tool in
                     ("segment_phases", "pulse_metrics") else None}
    return out, s


def report() -> None:
    s = Session.open(ROOT / "examples" / "hanaro_knsb" / "run.csv")
    res = RuleAgent("en").run(s)
    paths = write_report(s, res, HERE / "report_hanaro.html", "en")
    Path(paths["json"]).unlink(missing_ok=True)  # the demo links the HTML only


def main(argv: list[str]) -> None:
    led, _ = ledger()
    tpl = (HERE / "template.html").read_text()
    verifier = (HERE / "verifier.js").read_text()
    data = "window.HANARO_LEDGER = " + json.dumps(led, ensure_ascii=False) + ";"
    page = tpl.replace("/*{{VERIFIER}}*/", verifier).replace("/*{{LEDGER}}*/", data)
    if "--fragment" in argv:
        out = Path(argv[argv.index("--fragment") + 1])
        head = re.search(r"<head>(.*?)</head>", page, re.S).group(1)
        head = re.sub(r"<meta[^>]*>\s*", "", head)
        body = re.search(r"<body>(.*)</body>", page, re.S).group(1)
        out.write_text(head.strip() + "\n" + body.strip() + "\n")
        print(f"wrote {out}")
        return
    (HERE / "index.html").write_text(page)
    report()
    print(f"wrote {HERE / 'index.html'} and {HERE / 'report_hanaro.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
