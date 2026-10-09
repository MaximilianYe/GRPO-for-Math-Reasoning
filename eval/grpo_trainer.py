import os, re, json, math, argparse
import torch
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def train(model_path, rows, out_dir, lr=1e-4, num_epochs=1):

  tok = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
  
  lora_config = LoraConfig(
      r=16, lora_alpha=32, lora_dropout=0.05,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      task_type="CAUSAL_LM",
  )
  model.config.use_cache = False
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
  model.enable_input_require_grads()
  model = get_peft_model(model, lora_config) 

  model.print_trainable_parameters() 

  @torch.no_grad()
  def fill_old_logp(model, rows, tok, bs=4):
    model.eval()
    base = model.get_base_model()
    for i in range(0, len(rows), bs):
      chunk = rows[i:i+bs]
      b = collate([WeightedDataset(chunk)[j] for j in range(len(chunk))], tok.pad_token_id)
      b = {k: v.to("cuda") for k, v in b.items()}
      hidden = base.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
      logp, mask, row = token_logp(hidden, base.lm_head, b["labels"])
      for j, r in enumerate(chunk):
        r["old_logp"] = logp[row == j].tolist()
    model.train()


  fill_old_logp(model, rows, tok)

  optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
  
  loader = DataLoader(WeightedDataset(rows), batch_size=2, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
  ACCUM = 8
  
  model.train()
  
  for epoch in range(num_epochs):
    for step, batch in enumerate(loader):
      batch = {k: v.to("cuda") for k, v in batch.items()}
      
      base = model.get_base_model()
      hidden = base.model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).last_hidden_state
      loss = grpo_loss(hidden, base.lm_head, batch["labels"], batch["old_logp"], batch["A"])
      
      # backprop loss 
      (loss/ACCUM).backward()
      if (step + 1) % ACCUM == 0:
        
        # Update weights
        optimizer.step()
        # zero the parameter gradients
        optimizer.zero_grad()


      if step % 20 == 0:
        print(f"Step: {step}, Loss: {loss.item():.6f}")

    if (step + 1) % ACCUM != 0:
      optimizer.step()
      optimizer.zero_grad()
  
  model.save_pretrained(out_dir)
  tok.save_pretrained(out_dir)

  merged_dir = out_dir.replace("adapter_", "merged_")
  merged = model.merge_and_unload()
  merged.config.use_cache = True         
  merged.save_pretrained(merged_dir)
  tok.save_pretrained(merged_dir)
  print(f"merged -> {merged_dir}")


def collate(batch, pad_id):
   
    B = len(batch)
    L = max(len(b["input_ids"]) for b in batch)     

    ids = torch.full((B, L), pad_id, dtype=torch.long)   
    labels = torch.full((B, L), -100,   dtype=torch.long)   
    am = torch.zeros((B, L), dtype=torch.long)       
    old = torch.zeros((B, L), dtype=torch.float)   

    for i, b in enumerate(batch):
      n = len(b["input_ids"])
      ids[i, :n]    = torch.tensor(b["input_ids"])
      labels[i, :n] = torch.tensor(b["labels"])     
      am[i, :n]     = 1
      if b["old_logp"] is not None:
        P = n - len(b["old_logp"])          
        old[i, P:n] = torch.tensor(b["old_logp"])  
    A = torch.tensor([b["A"] for b in batch], dtype=torch.float)
    return dict(input_ids=ids, attention_mask=am, labels=labels, old_logp=old, A=A)


def load_train_rows(path):
  rows = [json.loads(l) for l in open(path)]
  return [r for r in rows if 0 < r["p"] < 1]      

class WeightedDataset(torch.utils.data.Dataset):
  def __init__(self, rows): self.rows = rows
  def __len__(self): return len(self.rows)
  def __getitem__(self, i):
    r = self.rows[i]
    return dict(input_ids=r["prompt_ids"] + r["completion_ids"],
                labels=[-100] * len(r["prompt_ids"]) + r["completion_ids"],
                A=r["A"], old_logp=r.get("old_logp"))


def token_logp(hidden, lm_head, labels, chunk=1024):
  h   = hidden[:, :-1]                 # [B, T-1, H]
  tgt = labels[:, 1:]
  mask = tgt != -100
  sel_h = h[mask]                      # [N, H]
  sel_t = tgt[mask]                    # [N]
  # [0, 1]  [[0],  [[0, 0, 0, 0, 0],  row = [0, 0, 0, 1, 1]  
  #       [1]]  [1, 1, 1, 1, 1]]
  row = torch.arange(h.size(0), device=h.device).unsqueeze(1).expand_as(mask)[mask]

  def ce(hc, tc):
    return -F.cross_entropy(lm_head(hc).float(), tc, reduction="none")
  logp = torch.cat([checkpoint(ce, sel_h[i:i+chunk], sel_t[i:i+chunk], use_reentrant=False)
                      for i in range(0, sel_h.size(0), chunk)])
  return logp, mask, row

def grpo_loss(hidden, lm_head, labels, old_logp, A, eps=0.2):
  logp, mask, row = token_logp(hidden, lm_head, labels)
  old = old_logp[:, 1:][mask]                  
  ratio = torch.exp(logp - old)
  a = A[row]                                  
  obj = torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
  B = hidden.size(0)
  seq = torch.zeros(B, device=hidden.device).index_add_(0, row, -obj)
  seq = seq / mask.sum(1).clamp(min=1)        
  return seq.mean() 

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  train_rows = load_train_rows(args.data)
  train(args.model, train_rows, args.out)