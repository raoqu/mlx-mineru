#!/usr/bin/env python3
"""Export the UniMERNet formula model (Swin encoder + mBART decoder) to ONNX.

  mfr_encoder.onnx : pixel_values[1,3,192,672] -> encoder_hidden[1,N,768]
  mfr_decoder.onnx : a *merged KV-cache* decoder graph (single weights copy, ~577MB):
      inputs : input_ids[1,1], attention_mask[1,L], encoder_hidden_states[1,N,768],
               self_past_0..15  (per layer: self-attn K[1,16,p,24], V[1,16,p,48])
      outputs: logits[1,1,50000], self_present_0..15
    Each of the 8 decoder layers caches its self-attention K/V (fed present->past every step),
    so a step processes only the new token (O(T) total instead of the old no-cache O(T^2) --
    ~3.7x on a 197-token formula, exact greedy ids). Cross-attention is recomputed from
    encoder_hidden each step (cheap, O(N)); we keep it OUT of the cache so a single traced graph
    handles both prefill (empty self past, seq 0) and step -- the cross-reuse `if` in the
    attention is data-dependent and would otherwise freeze to one branch at trace time.

Then validate ONNX KV greedy decode == torch greedy decode on a test formula.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MFR_DIR = os.path.join(HERE, "models", "pipeline", "MFR")
sys.path.insert(0, os.path.expanduser("~/research/MinerU"))
from mineru.model.mfr.unimernet.unimernet_hf import UnimernetModel  # noqa: E402

# opset-17 lacks aten::take_along_dim (used by some HF paths); alias to gather.
if not hasattr(torch, "_take_along_dim_orig"):
    torch._take_along_dim_orig = torch.take_along_dim
    torch.take_along_dim = lambda inp, idx, dim: torch.gather(inp, dim, idx)

LAYERS, HEADS, KDIM, VDIM = 8, 16, 24, 48  # mBART decoder cache dims (from config)


class EncoderWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc = m.encoder

    def forward(self, pixel_values):
        return self.enc(pixel_values).last_hidden_state


class DecoderKV(torch.nn.Module):
    """Merged KV-cache decoder: self-attn cached, cross-attn recomputed (empty cross past)."""

    def __init__(self, m):
        super().__init__()
        self.dec = m.decoder  # UnimerMBartForCausalLM

    def forward(self, input_ids, attention_mask, encoder_hidden_states, *self_past):
        # Per layer pass (self_k, self_v, empty_cross_k, empty_cross_v): the empty cross past
        # (seq 0) fails the attention's `past[0].shape[2] == N` check, so cross is recomputed.
        dt = encoder_hidden_states.dtype
        ek = torch.zeros(1, HEADS, 0, KDIM, dtype=dt)
        ev = torch.zeros(1, HEADS, 0, VDIM, dtype=dt)
        pkv = tuple((self_past[i * 2], self_past[i * 2 + 1], ek, ev) for i in range(LAYERS))
        out = self.dec(input_ids=input_ids, attention_mask=attention_mask,
                       encoder_hidden_states=encoder_hidden_states, past_key_values=pkv,
                       use_cache=True, return_dict=True)
        # keep only self present (positions 0,1 of each layer's 4-tuple)
        sp = tuple(t for layer in out.past_key_values for t in layer[:2])
        return (out.logits,) + sp


def main():
    m = UnimernetModel.from_pretrained(MFR_DIR, attn_implementation="eager").eval()
    enc = EncoderWrap(m).eval()
    dec = DecoderKV(m).eval()

    px = torch.zeros(1, 3, 192, 672, dtype=torch.float32)
    with torch.no_grad():
        hid = enc(px)
    print(f"encoder hidden: {tuple(hid.shape)}")

    enc_path = os.path.join(MFR_DIR, "mfr_encoder.onnx")
    dec_path = os.path.join(MFR_DIR, "mfr_decoder.onnx")
    torch.onnx.export(
        enc, (px,), enc_path, opset_version=17,
        input_names=["pixel_values"], output_names=["encoder_hidden"],
        dynamic_axes={"pixel_values": {0: "b"}, "encoder_hidden": {0: "b"}})
    print("wrote", enc_path)

    # KV-cache decoder: trace with a non-empty self past (seq 1) and matching attention_mask
    # length (past_seq + tgt = 2) so the self-attn concat branch is captured.
    pin = [f"self_past_{i}" for i in range(2 * LAYERS)]
    pout = [f"self_present_{i}" for i in range(2 * LAYERS)]
    self0 = []
    for _ in range(LAYERS):
        self0 += [torch.zeros(1, HEADS, 1, KDIM), torch.zeros(1, HEADS, 1, VDIM)]
    args = (torch.tensor([[59]]), torch.ones(1, 2, dtype=torch.long), hid, *self0)
    da = {"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "L"},
          "encoder_hidden_states": {0: "b", 1: "n"}, "logits": {0: "b", 1: "t"}}
    da.update({n: {0: "b", 2: "p"} for n in pin})
    da.update({n: {0: "b", 2: "p1"} for n in pout})
    torch.onnx.export(
        dec, args, dec_path, opset_version=17,
        input_names=["input_ids", "attention_mask", "encoder_hidden_states"] + pin,
        output_names=["logits"] + pout, dynamic_axes=da)
    print("wrote", dec_path, f"({os.path.getsize(dec_path) / 1e6:.0f} MB)")

    # --- parity: torch greedy (no cache) vs onnx KV greedy on a real formula ---
    import cv2
    import onnxruntime as ort
    fimg = "/tmp/formula.png"
    if not os.path.exists(fimg):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 1.2))
        fig.text(0.02, 0.4, r"$E=mc^2+\sum_{i=1}^{n}\frac{a_i}{b_i}+\int_0^1 x^2 dx$", fontsize=18)
        fig.savefig(fimg, dpi=120, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    img = cv2.cvtColor(cv2.imread(fimg), cv2.COLOR_BGR2RGB)
    t = m.transform(img).unsqueeze(0)
    with torch.no_grad():
        ref = m.generate({"image": t})
    ref_ids = list(ref["pred_ids"][0])

    es = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    hid = es.run(None, {"pixel_values": t.repeat(1, 3, 1, 1).numpy().astype(np.float32)})[0]
    EOS, BOS, MAXT = 2, 0, 1536
    sp = []
    for _ in range(LAYERS):
        sp += [np.zeros((1, HEADS, 0, KDIM), np.float32), np.zeros((1, HEADS, 0, VDIM), np.float32)]
    onnx_ids, cur = [], BOS
    for step in range(MAXT):
        feed = {"input_ids": np.array([[cur]], np.int64),
                "attention_mask": np.ones((1, step + 1), np.int64), "encoder_hidden_states": hid}
        for i in range(2 * LAYERS):
            feed[f"self_past_{i}"] = sp[i]
        out = ds.run(None, feed)
        cur = int(out[0][0, -1].argmax())
        onnx_ids.append(cur)
        if cur == EOS:
            break
        sp = out[1:]
    print(f"torch ids ({len(ref_ids)}):", ref_ids[:24])
    print(f"onnx  ids ({len(onnx_ids)}):", onnx_ids[:24])
    same = ref_ids == onnx_ids
    print("PARITY:", "EXACT" if same else f"DIFF (torch {len(ref_ids)} vs onnx {len(onnx_ids)})")


if __name__ == "__main__":
    main()
