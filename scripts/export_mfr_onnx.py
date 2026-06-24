#!/usr/bin/env python3
"""Export the UniMERNet formula model (Swin encoder + mBART decoder) to ONNX.

Two graphs (no KV cache -- formulas are short, recompute is simple and robust):
  mfr_encoder.onnx : pixel_values[1,3,192,672] -> encoder_hidden[1,N,768]
  mfr_decoder.onnx : input_ids[1,t] + encoder_hidden[1,N,768] -> logits[1,t,50000]

Then validate ONNX greedy decode == torch greedy decode on a test formula.
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


class EncoderWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc = m.encoder

    def forward(self, pixel_values):
        return self.enc(pixel_values).last_hidden_state


class DecoderWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.dec = m.decoder  # UnimerMBartForCausalLM

    def forward(self, input_ids, encoder_hidden_states):
        out = self.dec(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def main():
    m = UnimernetModel.from_pretrained(MFR_DIR, attn_implementation="eager").eval()
    enc = EncoderWrap(m).eval()
    dec = DecoderWrap(m).eval()

    px = torch.zeros(1, 3, 192, 672, dtype=torch.float32)
    with torch.no_grad():
        hid = enc(px)
    N = hid.shape[1]
    print(f"encoder hidden: {tuple(hid.shape)}")

    enc_path = os.path.join(MFR_DIR, "mfr_encoder.onnx")
    dec_path = os.path.join(MFR_DIR, "mfr_decoder.onnx")
    torch.onnx.export(
        enc, (px,), enc_path, opset_version=17,
        input_names=["pixel_values"], output_names=["encoder_hidden"],
        dynamic_axes={"pixel_values": {0: "b"}, "encoder_hidden": {0: "b"}})
    print("wrote", enc_path)

    ids = torch.tensor([[0, 59, 243, 51]], dtype=torch.long)
    torch.onnx.export(
        dec, (ids, hid), dec_path, opset_version=17,
        input_names=["input_ids", "encoder_hidden"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "t"}, "encoder_hidden": {0: "b", 1: "n"},
                      "logits": {0: "b", 1: "t"}})
    print("wrote", dec_path)

    # --- parity: torch greedy vs onnx greedy on a real formula ---
    import cv2
    import onnxruntime as ort
    fimg = "/tmp/formula.png"
    if not os.path.exists(fimg):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 1.2))
        fig.text(0.05, 0.4, r"$E=mc^2+\sum_{i=1}^{n}\frac{a_i}{b_i}$", fontsize=20)
        fig.savefig(fimg, dpi=120, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    img = cv2.cvtColor(cv2.imread(fimg), cv2.COLOR_BGR2RGB)
    t = m.transform(img).unsqueeze(0)            # [1,1,192,672]
    px = t.repeat(1, 3, 1, 1)
    with torch.no_grad():
        ref = m.generate({"image": t})
    ref_ids = list(ref["pred_ids"][0])

    es = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    hid = es.run(None, {"pixel_values": px.numpy().astype(np.float32)})[0]
    EOS, BOS, MAXT = 2, 0, 200
    seq = [BOS]
    for _ in range(MAXT):
        logits = ds.run(None, {"input_ids": np.array([seq], np.int64), "encoder_hidden": hid})[0]
        nxt = int(logits[0, -1].argmax())
        seq.append(nxt)
        if nxt == EOS:
            break
    onnx_ids = seq[1:]  # drop BOS; keeps trailing EOS, matching pred_ids convention
    print(f"torch ids ({len(ref_ids)}):", ref_ids[:24])
    print(f"onnx  ids ({len(onnx_ids)}):", onnx_ids[:24])
    same = ref_ids == onnx_ids
    print("PARITY:", "EXACT" if same else f"DIFF (torch {len(ref_ids)} vs onnx {len(onnx_ids)})")
    if not same:
        for i, (a, b) in enumerate(zip(ref_ids, onnx_ids)):
            if a != b:
                print(f"  first diff @ {i}: torch={a} onnx={b}")
                break


if __name__ == "__main__":
    main()
