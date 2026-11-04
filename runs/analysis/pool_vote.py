"""Vote over the samples of several runs of the same questions together (an ensemble of prompt conditions), no LLM:
python runs/analysis/pool_vote.py OUT RUN1 RUN2 ... [--threshold 0.5] [--split public_test]
A question some run judged outside the data dates (final with a reason) keeps that only when every run did."""
import importlib.util
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "train")]
spec = importlib.util.spec_from_file_location("predict", R / "sqlbase/predict.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from check_submission import problem  # noqa: E402

args = sys.argv[1:]
th, split = 0.5, "public_test"
if "--threshold" in args:
    i = args.index("--threshold"); th = float(args[i + 1]); del args[i:i + 2]
if "--split" in args:
    i = args.index("--split"); split = args[i + 1]; del args[i:i + 2]
out, runs = R / args[0], [R / x for x in args[1:]]
traces = [{t["id"]: t for t in map(json.loads, (r / "trace.jsonl").open(encoding="utf-8"))} for r in runs]
qs = [json.loads(line) for line in (R / "data" / split / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
results = {}
for q in qs:
    ts = [tr[q["id"]] for tr in traces if q["id"] in tr]
    if not ts:
        continue  # not asked in these runs (a train subset): abstain
    if ts and all(t["final"].get("reason") for t in ts):
        results[q["id"]] = ts[0]["final"]
        continue
    preds = [s["pred"] for t in ts for s in t["samples"]]
    ans, conf = p.vote(preds, len(preds))
    r = ans if p.decide(ans, conf, th) else {"type": "abstain"}
    results[q["id"]] = {"type": "abstain"} if problem({"id": q["id"], **r}) else r
out.mkdir(parents=True, exist_ok=True)
p.write_submission(out / "submission.jsonl", qs, results)
print(f"pooled {len(runs)} runs at threshold {th}: answered {sum(r['type'] != 'abstain' for r in results.values())} -> {out / 'submission.jsonl'}")
