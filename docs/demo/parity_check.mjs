// Check the JavaScript verifier against verdicts from the Python verifier (see make_parity_corpus.py).
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { verifyFinding } = require("./verifier.js");
const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
let nums = 0, bad = 0, badStatus = 0;
for (const c of cases) {
  const v = verifyFinding(c.finding, c.ledger);
  if (v.status !== c.status) badStatus++;
  const n = Math.max(v.numbers.length, c.numbers.length);
  for (let i = 0; i < n; i++) {
    nums++;
    const a = v.numbers[i], b = c.numbers[i];
    if (!a || !b || a.text !== b.text || a.grounded !== b.grounded || a.consistent !== b.consistent) {
      bad++;
      if (bad <= 10) console.log("MISMATCH", JSON.stringify({ js: a && [a.text, a.grounded, a.consistent, a.semanticProblem],
        py: b, statement: c.finding.statement.slice(0, 160) }));
    }
  }
}
console.log(`${cases.length} findings, ${nums} numbers: ${bad} number verdicts differ, ${badStatus} statuses differ`);
process.exit(bad || badStatus ? 1 : 0);
