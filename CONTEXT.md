# Groundline

Groundline analyses engine test data so that every number in its report traces back to a reproducible computation. An agent, rule-based or LLM, writes findings; a verifier checks them against the evidence.

## Language

### Runs and evidence

**Run**:
One test's recorded data (time plus channels), with its metadata, optional reference prediction and limits.
_Avoid_: test file, dataset

**Evidence**:
One recorded call of an analysis tool on a run: its parameters, result and provenance, with an ID such as `E4`.
_Avoid_: tool output, result entry

**Ledger**:
The ordered list of a run's evidence. Findings may only cite evidence in the ledger.
_Avoid_: log, history

### Claims and their checking

**Finding**:
One claim in a report (title, statement, category, severity, optional channel and time window), citing the evidence it rests on.
_Avoid_: conclusion, insight

**Source tag**:
A bracketed evidence field written right after a number (`742.3 K [E4.violations[0].peak_value]`), naming exactly where the number comes from.
_Avoid_: citation (a bare `[E4]` cites evidence; a source tag names a field)

**Verifier**:
The mechanical check of findings against the evidence they cite: numbers, units, role words, source tags, category and time window.
_Avoid_: validator, checker

**Verification**:
The verifier's verdict on one finding: verified, partial or unsupported, with what it made of each number and the problems it found.
_Avoid_: validation result

**Problem**:
Something about a finding as a whole that its cited evidence does not support, such as an unknown evidence ID or an anomaly category the evidence does not report. Problems about a single number's meaning are semantic problems instead.
_Avoid_: error, issue (an issue is a sensor-health result)
