#!/usr/bin/env python3
"""Reference: greedy-decode a FIXED random encoder hidden state through the ONNX mBART decoder
(merged KV-cache graph) and save (encoder, ids) so the MLX C++ decoder can be validated to
produce byte-identical token ids. Decodes a fixed number of steps (ignores EOS) to exercise the
per-step math + KV-cache accumulation thoroughly."""
import os, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(HERE, "mumodel/pipeline/MFR/mfr_decoder.onnx")
N, D, LAYERS, HEADS, QK, V, STEPS, BOS = 50, 768, 8, 16, 24, 48, 30, 0

rng = np.random.default_rng(0)
enc = rng.standard_normal((N, D)).astype(np.float32)
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
cache = {f"self_past_{2*i}": np.zeros((1, HEADS, 0, QK), np.float32) for i in range(LAYERS)}
cache.update({f"self_past_{2*i+1}": np.zeros((1, HEADS, 0, V), np.float32) for i in range(LAYERS)})
cur, ids = BOS, []
for step in range(STEPS):
    feeds = {"input_ids": np.array([[cur]], np.int64),
             "attention_mask": np.ones((1, step + 1), np.int64),
             "encoder_hidden_states": enc[None], **cache}
    out = sess.run(None, feeds)  # logits, self_present_0..15
    logits = out[0][0, 0]
    cur = int(logits.argmax()); ids.append(cur)
    for i in range(2 * LAYERS):
        cache[f"self_past_{i}"] = out[1 + i]

enc.tofile("/tmp/mfr_enc.f32")
open("/tmp/mfr_ref_ids.txt", "w").write(" ".join(map(str, ids)))
print(f"[ref] N={N} steps={STEPS} ids={ids}")
print("[ref] wrote /tmp/mfr_enc.f32 + /tmp/mfr_ref_ids.txt")
