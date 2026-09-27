import os, re, json, math, random, gc, time, subprocess, argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from math_verify import parse

import sys
sys.path.append('/content/drive/MyDrive/')
from verifier import score

INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def main(model, out, n_questions=128, k=8, seed=42):
  ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
  rng = random.Random(seed)
  idx = rng.sample(range(len(ds)), n_questions)
  batch = [dict(qid=i, problem=ds[i]["problem"], gold=ds[i]["solution"]) for i in idx]
  
  batch = [ex for ex in batch if parse(ex["gold"])]
  print(f"valid {len(batch)} / {n_questions} questions")

  llm = LLM(model=model, gpu_memory_utilization=0.85)
  sp = SamplingParams(n=k, temperature=1.0, top_p=1.0, max_tokens=2048, seed=seed)
  tok = AutoTokenizer.from_pretrained(model)
  prompt_texts = [
    tok.apply_chat_template(
        [{"role": "user", "content": ex["problem"] + "\n" + INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for ex in batch
  ]

  outs = llm.generate(prompt_texts, sp)

  rows = []
  im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
  for ex, o in zip(batch, outs):
    for c in o.outputs:
      rows.append(dict(qid=ex["qid"], gold=ex["gold"],
              prompt_ids=list(o.prompt_token_ids),
              completion_ids=list(c.token_ids),
              text=c.text, finish=c.finish_reason))
  
  # verifier
  n_patched = 0
  for row in rows:
    if row["finish"] == "stop" and row["completion_ids"][-1] != im_end_id:
      row["completion_ids"].append(im_end_id)
      n_patched += 1
    row["reward"] = score(row["gold"], row["text"], row["finish"])

  groups = defaultdict(list)
  for row in rows:
    groups[row["qid"]].append(row)
  
  for g in groups.values():
    p = sum(r["reward"] for r in g) / len(g)
    for r in g:
      r["p"] = p
      r["A"] = r["reward"] - p
      r["weight"] = max(r["A"], 0)      # r["A"]    
  with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

  ps = [g[0]["p"] for g in groups.values()]
  stats = dict(
    mean_reward=float(np.mean([r["reward"] for r in rows])),
    frac_informative=float(np.mean([0 < p < 1 for p in ps])),
    trunc_rate=float(np.mean([r["finish"] == "length" for r in rows])),
    mean_len=float(np.mean([len(r["completion_ids"]) for r in rows])),
    n_train=sum(r["weight"] > 0 for r in rows),
    n_patched=n_patched,
  )
  print("STATS " + json.dumps(stats))
  summary = dict(
    model=model, out=out, seed=seed, k=k,
    n_questions=n_questions, n_valid=len(batch), n_rows=len(rows),
    sampling=dict(temperature=1.0, top_p=1.0, max_tokens=2048),
    **stats,
  )
  summary_path = Path(out).with_suffix(".summary.json")
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
  print(f"summary -> {summary_path}")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n_questions", type=int, default=128)
  ap.add_argument("--k", type=int, default=8)
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  main(**vars(args))