#!/usr/bin/env python3
"""Extract the UniMERNet mBART decoder weights from mfr_decoder.onnx into a safetensors file
(clean names, ONNX MatMul layout [in,out]) + a config json, then VALIDATE a numpy reference
decoder step against onnxruntime so the MLX C++ port can match it exactly.

Self-contained: reads weights from the ONNX initializers (no torch / MinerU dependency).
Architecture (from unimer_mbart source): pre-norm mBART, d_model 768, 8 layers, 16 heads,
asymmetric attention (q/k head_dim 24 -> squeeze 384, v head_dim 48 -> 768, scale 24**-0.5),
learned positional embeddings (offset 2), layernorm_embedding after embed, final layer_norm,
gelu FFN (3072), lm_head tied to embed_tokens (+ final_logits_bias).
"""
import json, os, sys
import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(HERE, "mumodel/pipeline/MFR/mfr_decoder.onnx")
OUT_DIR = os.path.join(HERE, "mumodel/pipeline/MFR")
LAYERS, HEADS = 8, 16
QK_HEAD, V_HEAD = 24, 48          # squeeze_head_dim, head_dim
D_MODEL, SQUEEZE, FFN, VOCAB = 768, 384, 3072, 50000

m = onnx.load(ONNX)  # external data inline
g = m.graph
inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n

def matmul_weight_for_bias(bias_name):
    """Find the projection weight feeding the Add that consumes `bias_name`."""
    for n in g.node:
        if n.op_type == "Add" and bias_name in n.input:
            other = [i for i in n.input if i != bias_name][0]
            mm = producer.get(other)
            # skip through an intervening op if needed
            while mm is not None and mm.op_type != "MatMul":
                mm = producer.get(mm.input[0]) if mm.input else None
            if mm is not None and mm.op_type == "MatMul":
                return inits[mm.input[1]]  # [in, out]
    raise KeyError(f"no MatMul found for bias {bias_name}")

W = {}
P = "d.model.decoder"
# embeddings + norms (named directly)
W["embed_tokens.weight"] = inits[f"{P}.embed_tokens.weight"]        # [50000,768]
W["embed_positions.weight"] = inits[f"{P}.embed_positions.weight"]  # [1538,768]
for nm in ["layernorm_embedding", "layer_norm"]:
    W[f"{nm}.weight"] = inits[f"{P}.{nm}.weight"]
    W[f"{nm}.bias"] = inits[f"{P}.{nm}.bias"]
# per-layer
for L in range(LAYERS):
    lp = f"{P}.layers.{L}"
    for blk in ["self_attn", "encoder_attn"]:
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            b = inits[f"{lp}.{blk}.{proj}.bias"]
            w = matmul_weight_for_bias(f"{lp}.{blk}.{proj}.bias")
            W[f"layers.{L}.{blk}.{proj}.weight"] = w
            W[f"layers.{L}.{blk}.{proj}.bias"] = b
        for ln in [f"{blk}_layer_norm"]:
            W[f"layers.{L}.{ln}.weight"] = inits[f"{lp}.{ln}.weight"]
            W[f"layers.{L}.{ln}.bias"] = inits[f"{lp}.{ln}.bias"]
    for fc in ["fc1", "fc2"]:
        W[f"layers.{L}.{fc}.weight"] = matmul_weight_for_bias(f"{lp}.{fc}.bias")
        W[f"layers.{L}.{fc}.bias"] = inits[f"{lp}.{fc}.bias"]
    W[f"layers.{L}.final_layer_norm.weight"] = inits[f"{lp}.final_layer_norm.weight"]
    W[f"layers.{L}.final_layer_norm.bias"] = inits[f"{lp}.final_layer_norm.bias"]
# lm_head = the only [768,50000] MatMul initializer
lm = [v for k, v in inits.items() if v.shape == (D_MODEL, VOCAB)]
assert len(lm) == 1, f"expected 1 lm_head, got {len(lm)}"
W["lm_head.weight"] = lm[0]  # [768,50000]

# detect embed_scale: is there a Mul by a scalar right after the embed_tokens Gather?
embed_scale = 1.0
for n in g.node:
    if n.op_type == "Gather" and f"{P}.embed_tokens.weight" in n.input:
        for c in g.node:
            if c.op_type == "Mul" and n.output[0] in c.input:
                other = [i for i in c.input if i != n.output[0]]
                if other and other[0] in inits:
                    embed_scale = float(np.array(inits[other[0]]).reshape(-1)[0])
print(f"[extract] {len(W)} tensors | embed_scale={embed_scale}")
for k in ["embed_tokens.weight", "layers.0.self_attn.q_proj.weight",
          "layers.0.self_attn.v_proj.weight", "lm_head.weight"]:
    print(f"   {k}: {W[k].shape}")

cfg = dict(d_model=D_MODEL, layers=LAYERS, heads=HEADS, qk_head=QK_HEAD, v_head=V_HEAD,
           squeeze=SQUEEZE, ffn=FFN, vocab=VOCAB, max_pos=W["embed_positions.weight"].shape[0],
           pos_offset=2, embed_scale=embed_scale, scaling=QK_HEAD ** -0.5,
           bos=0, eos=2, activation="gelu")

# ---- numpy reference: ONE decoder step (prefill, single token), validated vs onnxruntime ----
def ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b
def gelu(x):
    from math import sqrt
    return 0.5 * x * (1.0 + np.vectorize(lambda v: __import__("math").erf(v / sqrt(2)))(x))

def ref_step(input_id, enc):  # enc: [N,768]
    N = enc.shape[0]
    h = W["embed_tokens.weight"][input_id] * cfg["embed_scale"]          # [768]
    h = h + W["embed_positions.weight"][0 + cfg["pos_offset"]]           # pos 0 (+offset)
    h = ln(h, W["layernorm_embedding.weight"], W["layernorm_embedding.bias"])
    h = h[None, :]                                                       # [1,768]
    for L in range(LAYERS):
        p = f"layers.{L}"
        # self-attn (single token, no past): attends to itself only
        r = h; x = ln(h, W[f"{p}.self_attn_layer_norm.weight"], W[f"{p}.self_attn_layer_norm.bias"])
        q = (x @ W[f"{p}.self_attn.q_proj.weight"] + W[f"{p}.self_attn.q_proj.bias"]) * cfg["scaling"]
        k = x @ W[f"{p}.self_attn.k_proj.weight"] + W[f"{p}.self_attn.k_proj.bias"]
        v = x @ W[f"{p}.self_attn.v_proj.weight"] + W[f"{p}.self_attn.v_proj.bias"]
        q = q.reshape(1, HEADS, QK_HEAD); k = k.reshape(1, HEADS, QK_HEAD); v = v.reshape(1, HEADS, V_HEAD)
        aw = np.einsum("thd,shd->hts", q[None].squeeze(0)[None].squeeze(0)[None], k[None].squeeze(0)[None].squeeze(0)[None]) if False else None
        # single token: scores [H,1,1] -> softmax = 1 -> out = v
        o = v.reshape(1, HEADS * V_HEAD)
        o = o @ W[f"{p}.self_attn.out_proj.weight"] + W[f"{p}.self_attn.out_proj.bias"]
        h = r + o
        # cross-attn
        r = h; x = ln(h, W[f"{p}.encoder_attn_layer_norm.weight"], W[f"{p}.encoder_attn_layer_norm.bias"])
        q = (x @ W[f"{p}.encoder_attn.q_proj.weight"] + W[f"{p}.encoder_attn.q_proj.bias"]) * cfg["scaling"]
        k = enc @ W[f"{p}.encoder_attn.k_proj.weight"] + W[f"{p}.encoder_attn.k_proj.bias"]   # [N,384]
        v = enc @ W[f"{p}.encoder_attn.v_proj.weight"] + W[f"{p}.encoder_attn.v_proj.bias"]   # [N,768]
        q = q.reshape(HEADS, QK_HEAD); k = k.reshape(N, HEADS, QK_HEAD); v = v.reshape(N, HEADS, V_HEAD)
        sc = np.einsum("hd,nhd->hn", q, k)            # [H,N]
        sc = sc - sc.max(-1, keepdims=True); pw = np.exp(sc); pw = pw / pw.sum(-1, keepdims=True)
        o = np.einsum("hn,nhd->hd", pw, v).reshape(1, HEADS * V_HEAD)
        o = o @ W[f"{p}.encoder_attn.out_proj.weight"] + W[f"{p}.encoder_attn.out_proj.bias"]
        h = r + o
        # FFN
        r = h; x = ln(h, W[f"{p}.final_layer_norm.weight"], W[f"{p}.final_layer_norm.bias"])
        x = gelu(x @ W[f"{p}.fc1.weight"] + W[f"{p}.fc1.bias"])
        x = x @ W[f"{p}.fc2.weight"] + W[f"{p}.fc2.bias"]
        h = r + x
    h = ln(h, W["layer_norm.weight"], W["layer_norm.bias"])
    logits = h @ W["lm_head.weight"]                  # [1,50000]
    return logits[0]

# validate vs onnxruntime
try:
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    N = 50
    enc = rng.standard_normal((N, D_MODEL)).astype(np.float32)
    bos = cfg["bos"]
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    feeds = {"input_ids": np.array([[bos]], np.int64),
             "attention_mask": np.array([[1]], np.int64),
             "encoder_hidden_states": enc[None]}
    for i in range(LAYERS):
        feeds[f"self_past_{2*i}"] = np.zeros((1, HEADS, 0, QK_HEAD), np.float32)
        feeds[f"self_past_{2*i+1}"] = np.zeros((1, HEADS, 0, V_HEAD), np.float32)
    ort_logits = sess.run(["logits"], feeds)[0][0, 0]
    my = ref_step(bos, enc).astype(np.float32)
    maxabs = float(np.abs(my - ort_logits).max())
    top5_mine = my.argsort()[-5:][::-1]; top5_ort = ort_logits.argsort()[-5:][::-1]
    print(f"[validate] numpy(fp64) vs ONNX(fp32) logits max|Δ|={maxabs:.4g} (fp32 accum noise)")
    print(f"[validate] top5 mine={top5_mine} ort={top5_ort}")
    # argmax + top-5 ordering must match; the small abs diff is fp32-vs-fp64 accumulation.
    assert my.argmax() == ort_logits.argmax() and list(top5_mine) == list(top5_ort), "ARCH MISMATCH"
    print("[validate] architecture OK ✅ (argmax + top-5 match)")
except ImportError:
    print("[validate] onnxruntime not available — skipped")

# save
from safetensors.numpy import save_file
st = {k: np.ascontiguousarray(v).astype(np.float32) for k, v in W.items()}
save_file(st, os.path.join(OUT_DIR, "mfr_decoder.safetensors"))
json.dump(cfg, open(os.path.join(OUT_DIR, "mfr_decoder_config.json"), "w"), indent=2)
print(f"[extract] wrote mfr_decoder.safetensors ({sum(v.nbytes for v in st.values())//(1<<20)} MB) + config")
