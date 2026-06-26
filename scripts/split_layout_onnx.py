#!/usr/bin/env python3
"""Split layout.onnx (PP-DocLayoutV2 / RT-DETR) so the conv backbone+encoder runs on MNN/Metal and
the Metal-hostile decoder (TopK/ScatterND/GridSample/CumSum deformable attention) stays on ORT.

Layout always runs at a fixed 800x800 (DetResizeForTest), so every backbone->decoder boundary
tensor EXCEPT the encoder memory is a shape-derived constant (anchors, spatial dims, valid ratios).
We bake those constants into the decoder and hand off the single real feature tensor:
  layout_backbone.onnx : pixel_values[1,3,800,800] -> memory[1,13125,256]   (Metal-clean -> MNN)
  layout_decoder.onnx  : memory -> logits, pred_boxes, order_logits          (ORT)
"""
import onnx, onnx.helper as H, onnx.numpy_helper as NP
import numpy as np, onnxruntime as ort
from collections import Counter, deque

SRC = "mumodel/pipeline/Layout/layout.onnx"        # full model (shipped baseline / fallback + source)
BB = "orgmodel/pipeline/Layout/layout_backbone.onnx"  # mnnconvert INTERMEDIATE (archived, not loaded)
BB_MNN = "mumodel/pipeline/Layout/layout_backbone.mnn"  # runtime (MNN/Metal backbone)
DEC = "mumodel/pipeline/Layout/layout_decoder.onnx"     # runtime (ORT decoder)
FINAL = ["logits", "pred_boxes", "order_logits"]
DIV = {"TopK", "CumSum", "ScatterND", "NonZero", "GridSample"}

m = onnx.load(SRC)
g = m.graph
inits = {i.name for i in g.initializer}
init_map = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
first = next(i for i, n in enumerate(g.node) if n.op_type in DIV)

def reachable(targets, stop=frozenset()):
    nodes, used_init, seen = [], set(), set()
    q = deque(targets); done = set()
    while q:
        t = q.popleft()
        if t in done: continue
        done.add(t)
        if t in stop: continue
        if t in inits: used_init.add(t); continue
        n = prod.get(t)
        if n and id(n) not in seen:
            seen.add(id(n)); nodes.append(n)
            for i in n.input: q.append(i)
    keep = {id(x) for x in nodes}
    return [n for n in g.node if id(n) in keep], used_init

import os
def build(path, name, nodes, used_init, ins, outs, extra_init=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gg = H.make_graph(nodes, name, ins, outs, [init_map[i] for i in used_init] + list(extra_init))
    mm = H.make_model(gg, opset_imports=m.opset_import); mm.ir_version = m.ir_version
    onnx.save(mm, path, save_as_external_data=False)

# boundary tensors (produced before `first`, consumed after, not initializers) that carry data
dep = {"pixel_values"}; changed = True
while changed:
    changed = False
    for n in g.node:
        if any(i in dep for i in n.input):
            for o in n.output:
                if o not in dep: dep.add(o); changed = True
pb = {o for i, n in enumerate(g.node) if i < first for o in n.output}
boundary = sorted({inp for i, n in enumerate(g.node) if i >= first
                   for inp in n.input if inp in pb and inp not in inits and inp in dep})

# find the single real-data tensor (varies with input content) vs constants (fixed at 800x800)
probe_nodes, probe_init = reachable(boundary)
build("/tmp/_probe.onnx", "probe", probe_nodes, probe_init, list(g.input),
      [H.make_empty_tensor_value_info(b) for b in boundary])
ps = ort.InferenceSession("/tmp/_probe.onnx", providers=["CPUExecutionProvider"])
x1 = np.random.rand(1, 3, 800, 800).astype(np.float32)
x2 = np.random.rand(1, 3, 800, 800).astype(np.float32)
o1 = ps.run(boundary, {"pixel_values": x1}); o2 = ps.run(boundary, {"pixel_values": x2})
real = [b for b, a, c in zip(boundary, o1, o2) if not (a.shape == c.shape and np.array_equal(a, c))]
const = [(b, a) for b, a, c in zip(boundary, o1, o2) if (a.shape == c.shape and np.array_equal(a, c))]
assert len(real) == 1, f"expected 1 real handoff tensor, got {real}"
MEM = real[0]
print(f"handoff (real): {MEM}  | baked constants: {len(const)}")

vi = {v.name: v for v in onnx.shape_inference.infer_shapes(m).graph.value_info}
mem_vi = vi[MEM]

# backbone: pixel_values -> MEM only
bb_nodes, bb_init = reachable([MEM])
build(BB, "layout_backbone", bb_nodes, bb_init, list(g.input), [mem_vi])
cb = Counter(n.op_type for n in bb_nodes)
print(f"backbone: {len(bb_nodes)} nodes Conv={cb.get('Conv',0)} divergent={ {k:cb[k] for k in DIV if cb.get(k)} or 'NONE'}")

# decoder: MEM (+baked consts) -> FINAL ; stop at MEM and the const tensors (now initializers)
const_inits = [NP.from_array(val, name=nm) for nm, val in const]
stop = frozenset([MEM] + [nm for nm, _ in const])
dec_nodes, dec_init = reachable(FINAL, stop=stop)
build(DEC, "layout_decoder", dec_nodes, dec_init, [mem_vi], [vi.get(o, H.make_empty_tensor_value_info(o)) for o in FINAL], const_inits)
cd = Counter(n.op_type for n in dec_nodes)
print(f"decoder: {len(dec_nodes)} nodes divergent={ {k:cd[k] for k in DIV if cd.get(k)} }")

# validate two-stage (ORT) == full (ORT)
full = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(FINAL, {"pixel_values": x1})
bbs = ort.InferenceSession(BB, providers=["CPUExecutionProvider"])
mem = bbs.run([MEM], {"pixel_values": x1})[0]
two = ort.InferenceSession(DEC, providers=["CPUExecutionProvider"]).run(FINAL, {MEM: mem})
print("=== two-stage vs full (ORT) ===")
for nm, a, b in zip(FINAL, full, two):
    print(f"  {nm}: max|Δ|={np.abs(a-b).max():.3g}")
print("HANDOFF_TENSOR =", MEM, "shape", mem.shape)

# convert the Metal-clean backbone (orgmodel intermediate) -> runtime MNN in mumodel.
import subprocess, shutil
if shutil.which("mnnconvert"):
    r = subprocess.run(["mnnconvert", "-f", "ONNX", "--modelFile", BB,
                        "--MNNModel", BB_MNN, "--bizCode", "mnn"], capture_output=True, text=True)
    print("mnnconvert:", "OK -> " + BB_MNN if r.returncode == 0 and "Success" in r.stdout else r.stderr[-300:])
else:
    print("mnnconvert not found — run it manually:", BB, "->", BB_MNN)
print(f"layout split: runtime={DEC} + {BB_MNN} (mumodel); intermediate={BB} (orgmodel)")
