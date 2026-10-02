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
  
  optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
  
  loader = DataLoader(WeightedDataset(rows), batch_size=2, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
  ACCUM = 8
  
  model.train()
  
  for epoch in range(num_epochs):
    for step, batch in enumerate(loader):
      batch = {k: v.to("cuda") for k, v in batch.items()}
      
      # # computing output
      # out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
      # logits = out.logits      
      # # evaluate loss
      # loss = loss_function(out.logits, batch["labels"], batch["weight"])
      base = model.get_base_model()
      hidden = base.model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).last_hidden_state
      loss = loss_function(hidden, base.lm_head, batch["labels"], batch["weight"])
      
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


def collate(batch, pad_id):
    B = len(batch)
    L = max(len(b["input_ids"]) for b in batch)       

    ids    = torch.full((B, L), pad_id, dtype=torch.long)   
    labels = torch.full((B, L), -100,   dtype=torch.long) 
    am     = torch.zeros((B, L), dtype=torch.long)         

    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n]    = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])     
        am[i, :n]     = 1

    w = torch.tensor([b["weight"] for b in batch], dtype=torch.float)
    return dict(input_ids=ids, attention_mask=am, labels=labels, weight=w)


def load_train_rows(path):
  rows = [json.loads(l) for l in open(path)]
  return [r for r in rows if r["weight"] > 0]      

class WeightedDataset(torch.utils.data.Dataset):
  def __init__(self, rows): self.rows = rows
  def __len__(self): return len(self.rows)
  def __getitem__(self, i):
    r = self.rows[i]
    return dict(input_ids=r["prompt_ids"] + r["completion_ids"],
                labels=[-100] * len(r["prompt_ids"]) + r["completion_ids"],
                weight=r["weight"])


def loss_function(hidden, lm_head, labels, w, chunk=1024): # loss_function(logits, labels, w):
  # logits = logits[:, :-1].float()
  # tgt = labels[:, 1:]
  # nll = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")
  # mask = (tgt != -100).float()
  # seq_nll = (nll * mask).sum(1) / mask.sum(1).clamp(min=1)   # avg token NLL for an answer in each batch

  # loss   = (w * seq_nll).sum() / w.shape[0]
  # return loss

  h   = hidden[:, :-1]                 # [B, T-1, H]
  tgt = labels[:, 1:]
  mask = tgt != -100
  sel_h = h[mask]                      # [N, H]
  sel_t = tgt[mask]                    # [N]
  row = torch.arange(h.size(0), device=h.device).unsqueeze(1).expand_as(mask)[mask]

  def ce(hc, tc):
      return F.cross_entropy(lm_head(hc).float(), tc, reduction="none")

  nll = torch.cat([
      checkpoint(ce, sel_h[i:i+chunk], sel_t[i:i+chunk], use_reentrant=False)
      for i in range(0, sel_h.size(0), chunk)
  ])

  seq_sum = torch.zeros(h.size(0), device=h.device).index_add_(0, row, nll)
  seq_nll = seq_sum / mask.sum(1).clamp(min=1)
  return (w * seq_nll).sum() / w.shape[0]

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  train_rows = load_train_rows(args.data)
  train(args.model, train_rows, args.out)