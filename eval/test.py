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
from vllm.lora.request import LoRARequest

import sys
sys.path.append('/content/drive/MyDrive/')
from verifier import score


INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."

def load_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [dict(qid=i, problem=x["problem"], gold="\\boxed{" + x["answer"] + "}")
            for i, x in enumerate(ds)]

def load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [dict(qid=i, problem=x["question"], gold="\\boxed{" + x["answer"].split("####")[-1].strip() + "}")
            for i, x in enumerate(ds)]

EVAL_CONFIGS = {
  "gsm8k_smoke": dict(loader=load_gsm8k,   temperature=0.0, top_p=1.0,  n=1, max_tokens=512),
  "math500":     dict(loader=load_math500, temperature=0.6, top_p=0.95, n=4, max_tokens=2048),
}

def evaluate(model_path, cfg_name, out_dir, seed, adapter=None):
  cfg = EVAL_CONFIGS[cfg_name]
  items = cfg["loader"]()          # [{qid, problem, gold}]
  if adapter:
    llm = LLM(model=model_path, gpu_memory_utilization=0.85,
              enable_lora=True, max_lora_rank=16)
    lora = LoRARequest("adapter", 1, adapter)
  else:
    llm = LLM(model=model_path, gpu_memory_utilization=0.85)
    lora = None

  sp = SamplingParams(n=cfg["n"], temperature=cfg["temperature"], top_p=cfg["top_p"],
                    max_tokens=cfg["max_tokens"], seed=seed)  
  tok = AutoTokenizer.from_pretrained(model_path)
  prompt_texts = [
    tok.apply_chat_template(
        [{"role": "user", "content": ex["problem"] + "\n" + INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for ex in items
  ]
  outs = llm.generate(prompt_texts, sp, lora_request=lora)

  rows = []
  for ex, o in zip(items, outs):
    for c in o.outputs:
      rows.append(dict(qid=ex["qid"], gold=ex["gold"], text=c.text, finish=c.finish_reason, n_tokens=len(c.token_ids), reward=score(ex["gold"], c.text, c.finish_reason)))
      
  groups = defaultdict(list)
  for r in rows:
    groups[r["qid"]].append(r["reward"])
  ps = [sum(g) / len(g) for g in groups.values()]
  metrics = dict(
    model=model_path, adapter=adapter, cfg_name=cfg_name, seed=seed,
    cfg={k: v for k, v in cfg.items() if k != "loader"},   
    n_questions=len(ps),
    mean_at_n=float(np.mean(ps)),                          
    pass_at_n=float(np.mean([p > 0 for p in ps])),         
    trunc_rate=float(np.mean([r["finish"] == "length" for r in rows])),
    mean_len=float(np.mean([r["n_tokens"] for r in rows])),
  )

  os.makedirs(out_dir, exist_ok=True)
  with open(f"{out_dir}/generations.jsonl", "w") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")
  with open(f"{out_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
  print("METRICS " + json.dumps(metrics))


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--adapter", default=None)
  ap.add_argument("--cfg", default="math500")
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  evaluate(args.model, args.cfg, args.out, args.seed, args.adapter)    