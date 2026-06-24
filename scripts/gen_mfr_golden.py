#!/usr/bin/env python3
"""Golden for the C++ formula recognizer. Renders a test formula, runs MinerU's ACTUAL
UniMERNet preprocess + the exported ONNX encoder/decoder (greedy) + byte-level decode.
Saves the preprocessed gray pixel map (isolates the transformer) AND the raw RGB crop
(for the C++ end-to-end preprocess test), plus expected token ids + LaTeX.
Also extracts the id->token vocab to mfr_vocab.txt for the C++ tokenizer.
"""
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MFR = os.path.join(HERE, "models", "pipeline", "MFR")
GOLDEN = os.path.join(HERE, "tests", "golden")
sys.path.insert(0, os.path.expanduser("~/research/MinerU"))
from mineru.model.mfr.unimernet.unimernet_hf import UnimernetModel  # noqa: E402

EOS, BOS, MAXT = 2, 0, 400


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
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
    ih, iw = img.shape[:2]
    np.ascontiguousarray(img).tofile(os.path.join(GOLDEN, "mfr_input.rgb"))

    m = UnimernetModel.from_pretrained(MFR, attn_implementation="eager").eval()
    gray = m.transform(img)[0].numpy().astype(np.float32)  # [192,672] normalized gray
    gray.tofile(os.path.join(GOLDEN, "mfr_pixel.f32"))
    H, W = gray.shape
    px = np.repeat(gray[None, None], 3, axis=1)  # [1,3,192,672]

    es = ort.InferenceSession(os.path.join(MFR, "mfr_encoder.onnx"), providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(os.path.join(MFR, "mfr_decoder.onnx"), providers=["CPUExecutionProvider"])
    hid = es.run(None, {"pixel_values": px})[0]
    N = hid.shape[1]
    seq = [BOS]
    for _ in range(MAXT):
        logits = ds.run(None, {"input_ids": np.array([seq], np.int64), "encoder_hidden": hid})[0]
        nxt = int(logits[0, -1].argmax())
        seq.append(nxt)
        if nxt == EOS:
            break
    ids = seq[1:]  # drop BOS, keep trailing EOS

    # byte-level decode (vocab only, no merges needed for decode)
    tok = json.load(open(os.path.join(MFR, "tokenizer.json")))
    inv = {v: k for k, v in tok["model"]["vocab"].items()}
    special = {0, 1, 2, 3}
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    s = "".join(inv[i] for i in ids if i not in special)
    latex = bytes(u2b[c] for c in s).decode("utf-8", errors="replace")

    # extract id->token vocab (one line per id, escaped) for the C++ tokenizer
    V = len(tok["model"]["vocab"])
    arr = [""] * V
    for t, i in tok["model"]["vocab"].items():
        arr[i] = t
    with open(os.path.join(MFR, "mfr_vocab.txt"), "w", encoding="utf-8") as f:
        for t in arr:
            f.write(t.replace("\\", "\\\\").replace("\n", "\\n") + "\n")

    out = {"pixel": "mfr_pixel.f32", "H": int(H), "W": int(W), "N": int(N),
           "input": "mfr_input.rgb", "src_w": int(iw), "src_h": int(ih),
           "ids": ids, "latex": latex}
    json.dump(out, open(os.path.join(GOLDEN, "mfr.json"), "w"), ensure_ascii=False, indent=1)
    print(f"wrote mfr.json: {len(ids)} ids, N={N}, latex:")
    print("  ", latex)


if __name__ == "__main__":
    main()
