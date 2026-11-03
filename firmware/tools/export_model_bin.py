"""Convert the int8 .tflite made by training/export_tflite.py into the firmware's own format.

    python tools/export_model_bin.py char_mlp_int8.tflite --ctx 256 --vocab 98 --verify

Why a new format? The firmware reads the first-layer table straight from flash, one row per input
position. A .tflite stores that matrix as [hidden][inputs], so one table row would not be contiguous.
Here it is transposed once, on the PC, to [inputs][hidden].

--verify runs random windows through the original .tflite (reference kernels) and compares the
hidden layer with the Python twin of the firmware code. They must match exactly.
"""
import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cmlp  # noqa: E402


def make_interpreter(path):
    """Interpreter that keeps intermediate tensors and uses the plain reference kernels."""
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        return Interpreter(model_path=path, experimental_preserve_all_tensors=True,
                           experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    except ImportError:
        import tensorflow as tf
        try:
            return tf.lite.Interpreter(
                model_path=path, experimental_preserve_all_tensors=True,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        except TypeError:
            return tf.lite.Interpreter(model_path=path, experimental_preserve_all_tensors=True)


def per_channel(scales, n):
    scales = np.asarray(scales, dtype=np.float64).reshape(-1)
    return np.full(n, scales[0]) if scales.size == 1 else scales


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tflite")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "model" / "model.bin"))
    ap.add_argument("--ctx", type=int, default=256, help="window length used in training")
    ap.add_argument("--vocab", type=int, default=98, help="vocabulary size used in training")
    ap.add_argument("--verify", action="store_true", help="compare with the .tflite on random windows")
    a = ap.parse_args()

    it = make_interpreter(a.tflite)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    fcs = [i for i, o in enumerate(ops) if o["op_name"] == "FULLY_CONNECTED"]
    if len(fcs) != 2:
        sys.exit(f"expected 2 FULLY_CONNECTED ops, found {len(fcs)}: {[o['op_name'] for o in ops]}")
    between = [o["op_name"] for o in ops[fcs[0] + 1:fcs[1]]]
    if any(n in ("RELU", "RELU6") for n in between):
        sys.exit("ReLU is a separate op in this .tflite; the firmware expects it fused into the first Dense")

    fc1, fc2 = ops[fcs[0]], ops[fcs[1]]
    x_idx, w1_idx, b1_idx = fc1["inputs"][:3]
    hid_idx = fc1["outputs"][0]
    w2_idx, b2_idx = fc2["inputs"][1], fc2["inputs"][2]

    W1 = it.get_tensor(w1_idx)            # int8 [hidden, ctx*vocab]
    W2 = it.get_tensor(w2_idx)            # int8 [vocab_out, hidden]
    b1 = it.get_tensor(b1_idx).astype(np.int64)
    b2 = it.get_tensor(b2_idx).astype(np.int64)
    if W1.dtype != np.int8 or W2.dtype != np.int8:
        sys.exit("weights must be int8")
    H, LV = W1.shape
    V2 = W2.shape[0]
    if LV != a.ctx * a.vocab:
        sys.exit(f"W1 has {LV} inputs but --ctx {a.ctx} x --vocab {a.vocab} = {a.ctx * a.vocab}")
    if W2.shape[1] != H:
        sys.exit("W2 does not match W1")
    for idx, name in ((w1_idx, "W1"), (w2_idx, "W2")):
        zp = np.asarray(td[idx]["quantization_parameters"]["zero_points"])
        if zp.size and np.any(zp != 0):
            sys.exit(f"{name} has a non-zero zero point")

    in_s, in_zp = td[x_idx]["quantization"]
    in_s = float(np.float32(in_s))
    h_s, h_zp = td[hid_idx]["quantization"]
    h_s = float(np.float32(h_s))
    s1 = per_channel(td[w1_idx]["quantization_parameters"]["scales"], H).astype(np.float32).astype(np.float64)
    s2 = per_channel(td[w2_idx]["quantization_parameters"]["scales"], V2).astype(np.float32).astype(np.float64)

    # input: 0.0 -> zero point, 1.0 -> q_one. The contribution of the zeros cancels, leaving in_delta per active symbol.
    q_one = min(127, int(math.floor(1.0 / in_s + 0.5)) + int(in_zp))
    in_delta = q_one - int(in_zp)
    act_min, act_max = max(int(h_zp), -128), 127      # ReLU fused into the first Dense

    qmul = np.zeros(H, dtype=np.int64)
    qshift = np.zeros(H, dtype=np.int64)
    for o in range(H):
        qmul[o], qshift[o] = cmlp.quantize_multiplier(in_s * s1[o] / h_s)
    if qshift.min() < -31 or qshift.max() > 30:
        sys.exit("requantisation shift out of range")
    scale2 = (np.float64(np.float32(h_s)) * s2).astype(np.float32)

    model = cmlp.CmlpModel(a.ctx, a.vocab, H, V2, in_delta, int(h_zp), act_min, act_max,
                           W1.T.copy(), W2, b1, qmul, qshift, b2, scale2)
    blob = cmlp.pack(model)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_bytes(blob)
    print(f"wrote {a.out}: {len(blob) / 2**20:.2f} MiB  (ctx {a.ctx}, vocab {a.vocab}, hidden {H}, out {V2})")
    print(f"  input scale {in_s:.6g} zp {in_zp}  in_delta {in_delta}   hidden scale {h_s:.6g} zp {h_zp}")

    if a.verify:
        inp = it.get_input_details()[0]
        shape = tuple(inp["shape"])
        rng = np.random.default_rng(1)
        worst = 0
        for _ in range(20):
            w = rng.integers(0, a.vocab, a.ctx)
            x = np.full(a.ctx * a.vocab, int(in_zp), dtype=np.int8)
            x[np.arange(a.ctx) * a.vocab + w] = q_one
            it.set_tensor(inp["index"], x.reshape(shape))
            it.invoke()
            ref = it.get_tensor(hid_idx).reshape(-1).astype(np.int32)
            mine, _ = cmlp.predict(model, w)
            worst = max(worst, int(np.abs(ref - mine.astype(np.int32)).max()))
        print(f"verify: largest difference of the hidden layer vs the .tflite = {worst} (must be 0)")
        if worst != 0:
            sys.exit("verification FAILED - do not flash this model")


if __name__ == "__main__":
    main()
