#!/usr/bin/env python3
"""Generate tokenizer golden using the REAL HF transformers tokenizer for the
model. Freezes encode()/decode() expectations into tests/golden/tokenizer.json."""
import json
import os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.path.join(HERE, "models", "MinerU2.5-tokenizer")
OUT = os.path.join(HERE, "tests", "golden", "tokenizer.json")

tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)

# A corpus exercising: ASCII words, contractions, CJK, digits (split per-char),
# punctuation, multiple spaces, newlines, mixed, and special tokens / chat markers.
CORPUS = [
    "Hello, world!",
    "It's a test, don't you think? We'll see.",
    "公式如 E = mc^2 所示，这是中文段落。",
    "MinerU 2.5 parses PDFs into Markdown.",
    "Numbers: 1234567890 and 42 things.",
    "Multiple   spaces\tand\ttabs.",
    "line one\nline two\n\nparagraph.",
    "Mixed 中文 and English 123 测试.",
    "  leading and trailing  ",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nLayout Detection:<|im_end|>\n"
    "<|im_start|>assistant\n",
    "Special: <|endoftext|> and <|object_ref_start|> markers.",
    "Table Recognition:\nFormula Recognition:\nText Recognition:",
]

cases = []
for s in CORPUS:
    ids = tok.encode(s, add_special_tokens=False)
    cases.append({
        "text": s,
        "ids": ids,
        "decoded": tok.decode(ids, skip_special_tokens=False),
        "decoded_skip": tok.decode(ids, skip_special_tokens=True),
    })

out = {
    "model": "opendatalab/MinerU2.5-Pro-2605-1.2B",
    "transformers": __import__("transformers").__version__,
    "vocab_size": tok.vocab_size,
    "cases": cases,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote", OUT, f"({len(cases)} cases)")
for c in cases[:4]:
    print(f"  {c['text'][:30]!r:34s} -> {len(c['ids'])} ids: {c['ids'][:12]}")
