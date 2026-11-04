"""QLoRA on Gemma 4 12B with Unsloth, for the 24 GB dev GPU. Loss on the reply only (thought + code).

    .venv-train/bin/python train/train_lora.py data/sft/rest.jsonl --out runs/lora/r16 [--max-steps 20]

Each line of the data files: {"system", "user", "reply"} (train/build_sft.py). The prompt is rendered with the
model's own chat template and enable_thinking=True, exactly as 02_agent sends it to vLLM, so the adapter learns
the continuation the agent will ask for at inference.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from unsloth import FastModel  # noqa: I001  (unsloth patches transformers on import: keep it first)
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

END = "<turn|>"   # Gemma 4 closes the model turn with it (eos id 106); Qwen: --end '<|im_end|>'


def load_rows(paths: list[Path], max_len: int, tokenizer, thinking: bool = True, end: str = END) -> list[dict]:
    rows, dropped = [], 0
    for p in paths:
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=r.get("thinking", thinking))
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(r["reply"] + end, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_len:
                dropped += 1
                continue
            rows.append({"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids})
    print(f"{len(rows)} examples, {dropped} longer than {max_len} tokens dropped")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap for a quick throughput / memory check")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--init-adapter", type=Path, help="continue from this adapter's weights (same r and target modules)")
    ap.add_argument("--end", default=END, help="end-of-turn text appended to every reply ('<|im_end|>' for Qwen)")
    ap.add_argument("--no-thinking", action="store_true", help="render prompts with enable_thinking=False (non-thinking replies)")
    args = ap.parse_args()

    model, tokenizer = FastModel.from_pretrained(model_name=args.model, max_seq_length=args.max_len,
                                                 load_in_4bit=True, full_finetuning=False)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.r, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        random_state=args.seed, use_gradient_checkpointing="unsloth")
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(args.init_adapter / "adapter_model.safetensors"))
        print(f"continuing from {args.init_adapter}: {len(res.unexpected_keys)} unexpected keys")
    tok = getattr(tokenizer, "tokenizer", tokenizer)   # the processor wraps the text tokenizer
    rows = load_rows(args.data, args.max_len, tok, thinking=not args.no_thinking, end=args.end)
    random.Random(args.seed).shuffle(rows)
    n_tok = sum(len(r["input_ids"]) for r in rows)
    n_lab = sum(sum(x != -100 for x in r["labels"]) for r in rows)
    print(f"tokens: {n_tok:,} total, {n_lab:,} trained ({n_lab / max(n_tok, 1):.1%})")

    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=str(args.out), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, warmup_steps=max(1, int(0.05 * args.max_steps)) if args.max_steps > 0 else 10,
            lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.0, bf16=True, logging_steps=1,
            save_strategy="steps", save_steps=50, save_total_limit=2, report_to="none", seed=args.seed,
            remove_unused_columns=False),
    )
    t = time.time()
    trainer.train()
    dt = time.time() - t
    steps = trainer.state.global_step
    print(f"{steps} steps in {dt:.0f}s; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; "
          f"~{n_tok * (steps * args.grad_accum / max(len(rows), 1)) / dt:,.0f} tokens/s")
    model.save_pretrained(str(args.out / "adapter"))
    tok.save_pretrained(str(args.out / "adapter"))


if __name__ == "__main__":
    main()
