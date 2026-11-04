# DataMate

Answers Vietnamese business questions over each shop's CSV/XLSX/SQLite exports (Zalo AI Challenge 2026). Scoring is +1 correct, −1 wrong, 0 abstain, so the system answers only when its sampled programs agree.

## Method

1. **Load** (`ingest.py`): every sheet/table → typed DuckDB table with its original column names. Generic readers for the export formats in the data: `1.234.000`, `198 000`, `55.000đ`, `25k`, `8tr600`, `1,27tr`; `dd/mm/yyyy`, `25 thg 2 2026`, `3/6` (year from sheet/file name, rulebook §8), Excel serials; `2 gói` → quantity + unit column. Rulebook §8 is applied at load: blank order-level cells on continuation lines take the order's first-line value, identical lines of one order count once. A status/note column gets BOOLEAN flags `_huy` / `_tra` (row is cancelled / returned) and, with an order id, `_don_huy` / `_don_tra` (some line of the order is), so every sampled query applies the same status rule. Footer/subtotal rows are dropped; same-layout monthly tables are unioned into `*_all` views.
2. **Prompt** (`predict.py`): system = SQL rules + full split rulebook (shared prefix, prefix-cached); user = README + schema with value lists/ranges + date anchors computed from `as_of_date` + exact data values matching the question + question.
3. **Sample** K DuckDB queries per question (Qwen3.5-9B, non-thinking, temperature 0.7), K/2 from each of two system prompts that word the status/name/date rules differently (multi-path self-consistency); execute each read-only with a timeout; a failed sample gets one greedy repair with the error message.
4. **Vote** on the executed, type-formatted answers. Answer if the top answer has ≥ `--threshold` (default 6/16) of the K votes, is not `unanswerable`, and is not 0 / empty for a number, percent, list or table (train gold has 4 such answers in 2,458; the vote was 0/63 correct on them); otherwise abstain.

No answers, business IDs or question templates are hard-coded; inference never reads `answers.jsonl`.

## Results (train dev300: 10 questions × 30 train businesses, `random.Random(0)`)

| Version | Vote accuracy | PA @0.5 | Best PA (threshold) |
| --- | ---: | ---: | ---: |
| v1 loader v1, K=8 | 46.7% | +0.140 | +0.163 (0.75) |
| v3 + dirty-format readers, §8 rules, prompt rules, K=8 | 60.7% | +0.337 | +0.337 (0.375–0.625) |
| v5 longer prompt rules, K=16 | – | +0.340 | +0.363 (0.375) |
| two-prompt mix (v3 + v5 rules), 8+8 samples | 64.7% | +0.410 | +0.433 (0.375) |
| + abstain on 0/empty answers | 64.7% | +0.437 | +0.470 (0.375) |
| + single-quote double-quoted strftime formats at execution | 66.0% | +0.467 | +0.497 (0.375) |
| **v6: + status flags at load, quarter windows in date anchors (real run)** | 72.3% | +0.547 | **+0.600 (0.375)** |

v6 vs previous, paired bootstrap on the same 300 questions: +0.10 at 0.375 (95% CI 0.037–0.163). v6 takes ~77 min per 300 questions on one A5000 (fewer repairs).

Mix rows are simulated from the v3 and v5 traces (same loader); seed-to-seed noise is about ±0.025.

Decision rule checked with 5-fold cross-validation grouped by business: on the mix trace the CV-tuned share threshold scores 0.423, and the 0/empty rule raises it to 0.463. A logistic-regression decision model over 16 vote features reaches 0.483 / 0.443 / 0.383 against the rule's 0.463 / 0.440 / 0.383 on three traces, so it was not adopted. Thresholds are chosen on dev300 (in-sample); public/private businesses are unseen and include new industries, so expect lower PA. Single A5000, fp8: ~2,650 s per 300 questions with K=8 and ~4,500 s with K=16, including ~330 s loading.

## Run

```bash
python predict.py --data /path/to/split --output out/ --model models/Qwen3.5-9B --k 16 --tp 2
python evaluation.py --data train --pred out/submission.jsonl --questions dev300.jsonl --output score.json
python evaluation.py --data train --trace out/trace.jsonl --output sweep.json   # PA by vote-share threshold
```

Outputs: `submission.jsonl` (`id`, `type`, `answer`; written as all-abstain first so a killed run is still valid), `trace.jsonl` (all samples, SQL, errors, votes), `run.json`.

Model: Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (9.65B params, Apache-2.0), see `model_manifest.json`. Seed 0.

### Run the 0.491 setup

From the repo root, after installing `sqlbase/requirements.txt` and putting the public test data in `data/public_test`:

```bash
git lfs install && git lfs pull   # runs/lora/sql_rft_v2/adapter is stored in Git LFS
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir sqlbase/models/Qwen3.5-9B
bash runs/analysis/run_v2_public.sh runs/my_run   # about 2.1 h on one 24 GB GPU; the file to submit is runs/my_run_debtroute/submission.jsonl
```

The script uses LoRA v2, `sqlbase/fewshot_bank24.jsonl` and the views and debt layers saved from that run (`sqlbase/config_v2_public`), then applies debt routing. Sampling differs between runs and machines, so the score will not match 0.491 exactly. The bank and the saved layers are built from competition data: keep them inside the team.

## Docker

```bash
docker build -t datamate .
docker run --rm --gpus all --network none -v /abs/split:/data:ro -v "$PWD/result:/result" datamate
```

`predict.sh` defaults: `/data`, `/result`, `/models/Qwen3.5-9B`, `--tp 2`, 5h58m timeout.
