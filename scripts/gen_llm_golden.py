#!/usr/bin/env python3
"""Golden for the Qwen2-VL language-model forward, using the REAL transformers
model (text-only). Freezes next-token argmax + top-k logits and a short greedy
continuation so the MLX C++ LLM forward can be verified.

Note: bf16 matmul ordering differs between torch and MLX, so logit VALUES are
compared with tolerance; the argmax (greedy decode) must match exactly."""
import json
import os
import sys

import torch
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models", "MinerU2.5-tokenizer")
OUT = os.path.join(HERE, "tests", "golden", "llm.json")

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16
).eval()

PROMPTS = [
    "Text Recognition:",
    "The quick brown fox",
    "MinerU converts PDF documents into",
]

cases = []
with torch.no_grad():
    for s in PROMPTS:
        ids = tok.encode(s, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long)
        logits = model(input_ids=inp).logits[0, -1].float()
        topv, topi = torch.topk(logits, 10)
        # 5-step greedy continuation, recording each step's top-2 gap so the C++
        # test can tolerate bf16 near-tie flips (gap small) but catch real bugs.
        cont = []
        cur = list(ids)
        for _ in range(5):
            x = torch.tensor([cur], dtype=torch.long)
            lg = model(input_ids=x).logits[0, -1].float()
            t2 = torch.topk(lg, 2).values.tolist()
            nxt = int(lg.argmax())
            cont.append({"token": nxt, "gap": float(t2[0] - t2[1])})
            cur.append(nxt)
        cases.append({
            "text": s,
            "ids": ids,
            "argmax": int(logits.argmax()),
            "top10": [[int(i), float(v)] for v, i in zip(topv.tolist(), topi.tolist())],
            "greedy5": cont,
        })

out = {"model": "MinerU2.5-Pro-2605-1.2B", "cases": cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", OUT)
for c in cases:
    print(f"  {c['text'][:30]!r:32s} ids={len(c['ids'])} argmax={c['argmax']} greedy5={c['greedy5']}")
