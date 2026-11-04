#!/usr/bin/env bash
# One predict.py process per GPU (data parallel over shops), then merge into /result/submission.jsonl + /result/logs/.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
DATA="${DATA_DIR:-/data}" OUT="${RESULT_DIR:-/result}"
NG="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
mkdir -p "$OUT/logs"
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i timeout --signal=TERM --kill-after=15s "${TIMEOUT_SECONDS:-21000}s" \
    python3 predict.py --data "$DATA" --output "$OUT/shard$i" --shard "$i/$NG" \
    --model "${MODEL_DIR:-/models/Qwen3-14B}" --quant "${QUANT:-fp8}" --k 8 --think-k 4 --deadline "$(( ${TIMEOUT_SECONDS:-21000} - 900 ))" --gpu-mem 0.92 \
    --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096}' "$@" > "$OUT/logs/_shard$i.txt" 2>&1 &
done
wait
python3 - "$DATA" "$OUT" "$NG" <<'EOF'
import json, shutil, sys
from pathlib import Path
data, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
got = {}
for i in range(n):
    s = out / f"shard{i}"
    if (s / "submission.jsonl").exists():
        got.update((r["id"], r) for r in map(json.loads, (s / "submission.jsonl").read_text(encoding="utf-8").splitlines()))
    for f in (s / "logs").glob("*.json") if (s / "logs").exists() else []:
        shutil.copy(f, out / "logs" / f.name)
ids = [json.loads(l)["id"] for l in (data / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
with (out / "submission.jsonl").open("w", encoding="utf-8") as f:
    for i in ids:
        f.write(json.dumps(got.get(i, {"id": i, "type": "abstain"}), ensure_ascii=False) + "\n")
print(f"merged {sum(i in got for i in ids)}/{len(ids)} questions -> {out / 'submission.jsonl'}")
EOF
