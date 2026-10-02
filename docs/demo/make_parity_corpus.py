"""Write a corpus of findings + evidence + Python verifier verdicts, for parity_check.mjs.

    python docs/demo/make_parity_corpus.py corpus.json [n_runs]
    node docs/demo/parity_check.mjs corpus.json
"""

import json
import random
import sys

from groundline.agent import RuleAgent
from groundline.findings import verify_finding
from groundline.session import Session
from groundline.synth import generate_run
from groundline.verifier_bench import _mutate, _tagged


def case(f, s):
    v = verify_finding(f, s)
    ledger = {e: {"result": s.evidence(e).result, "params": s.evidence(e).params}
              for e in f.evidence if s.evidence(e) is not None}
    return {"finding": {"title": f.title, "statement": f.statement, "evidence": f.evidence}, "ledger": ledger,
            "status": v["status"], "numbers": [{"text": n["text"], "grounded": n["grounded"],
                                                "consistent": n["consistent"]} for n in v["numbers"]]}


def main(out, n=20):
    rng = random.Random(0)
    cases = []
    for lang in ("zh", "en"):
        for i in range(n):
            run = generate_run(1000 + i)
            s = Session(run.data, run.meta, run.reference, run.limits)
            for f in RuleAgent(lang).run(s).findings:
                for g in (f, _tagged(f, s)):  # as written, and with every number's source field tagged
                    cases.append(case(g, s))
                    cases += [case(h, s) for _, h, _ in _mutate(g, s, rng)]
    with open(out, "w") as fh:
        json.dump(cases, fh, ensure_ascii=False)
    print(f"{len(cases)} cases, {sum(len(c['numbers']) for c in cases)} numbers -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
