"""export_tflite.py - float Keras model  ->  int8 .tflite (with calibration) + quality check.

  python training/export_tflite.py --run training/runs/qa3000 --data training/data/qa3000 \
         --out training/runs/qa3000/char_mlp_int8.tflite

Steps:
 1. load the trained model and copy its weights into an ordinary  Dense -> ReLU -> Dense  "twin"
    (the lookup table of PositionalOneHotDense is exactly the Dense kernel for a one-hot input);
 2. check that twin and original give the same numbers;
 3. calibrate: run REAL windows of text through the twin so the converter learns the value ranges
    of every layer, then convert to int8;
 4. generate answers with the float model and with the int8 model and compare exact matches.
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import END, Examples, load_pairs, load_vocab  # noqa: E402
from train import build_model, generate  # noqa: E402

keras = tf.keras


def one_hot_flat(x, V):
    """(n, W) symbol indices -> (n, W*V) float32 one-hot; column = position*V + symbol."""
    n, W = x.shape
    out = np.zeros((n, W * V), dtype=np.float32)
    out[np.arange(n)[:, None], np.arange(W)[None, :] * V + x] = 1.0
    return out


def build_twin(model, W, V, H, clip=None):
    """Plain Keras model with the same weights. Ends with logits (no softmax): the board only needs argmax.
    clip=(lo, hi): logits are clipped to this range before the output. Symbols whose logit is far below the
    best one can never be chosen, but they stretch the int8 range (here -700..+70 -> step of 3 logit units),
    so clipping to the useful top region gives a much finer int8 step."""
    table = model.layers[1].table.numpy()          # (W*V, H) = Dense kernel for one-hot input
    b1 = model.layers[1].bias.numpy()
    W2, b2 = model.layers[3].get_weights()
    inp = keras.Input(shape=(W * V,))
    h = keras.layers.Dense(H, activation="relu")(inp)
    out = keras.layers.Dense(V)(h)
    if clip is not None:
        lo, hi = clip
        out = keras.layers.Lambda(lambda t: tf.clip_by_value(t, lo, hi))(out)
    twin = keras.Model(inp, out)
    twin.layers[1].set_weights([table, b1])
    twin.layers[2].set_weights([W2, b2])
    return twin


def calibration_windows(data_dir, stoi, W, n, seed=0):
    """n random REAL windows: question end, middle of answers, END... the ranges must look like real use."""
    pairs = load_pairs(os.path.join(data_dir, "train.jsonl"), per_article=2)
    ex = Examples(pairs, stoi, W)
    ids = np.random.default_rng(seed).choice(len(ex), size=min(n, len(ex)), replace=False)
    return np.stack([ex.window_at(*ex.index[i]) for i in ids])


def generate_tflite(interp, question, stoi, vocab, W, V, max_len=400):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    seq = [stoi.get(c, stoi[" "]) for c in question + "\n"]
    res = []
    for _ in range(max_len):
        w = seq[-W:]
        x = np.array([[0] * (W - len(w)) + w], dtype=np.int32)
        q = np.clip(np.round(one_hot_flat(x, V) / s_in + z_in), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        nxt = int(np.argmax(interp.get_tensor(out["index"])[0]))
        if vocab[nxt] == END:
            break
        res.append(vocab[nxt])
        seq.append(nxt)
    return "".join(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with last.weights.h5")
    ap.add_argument("--data", required=True, help="folder with train.jsonl and vocab.json")
    ap.add_argument("--out", required=True, help="output .tflite path")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=480)
    ap.add_argument("--calib", type=int, default=1000, help="number of calibration windows")
    ap.add_argument("--check", type=int, default=100, help="questions for the float-vs-int8 comparison")
    ap.add_argument("--no-clip", action="store_true", help="do not clip the logits (reproduces the coarse output)")
    ap.add_argument("--clip-margin", type=float, default=20.0,
                    help="how far below the lowest best-logit the clipping range starts")
    a = ap.parse_args()
    W, H = a.window, a.hidden

    vocab, stoi = load_vocab(os.path.join(a.data, "vocab.json"))
    V = len(vocab)
    model = build_model(W, V, H)
    model.load_weights(os.path.join(a.run, "last.weights.h5"))
    calib = calibration_windows(a.data, stoi, W, a.calib)

    # useful range of logits = where the BEST symbol lives; everything far below it can be clipped
    logits = np.concatenate([model(calib[i:i + 200]).numpy() for i in range(0, len(calib), 200)])
    top = logits.max(axis=1)
    clip = None
    if not a.no_clip:
        clip = (float(top.min() - a.clip_margin), float(top.max() + 5.0))
        print(f"best-logit range on calibration windows: {top.min():.1f} .. {top.max():.1f}; "
              f"clipping logits to [{clip[0]:.1f}, {clip[1]:.1f}]")
    twin = build_twin(model, W, V, H, clip)

    ref = np.clip(logits[:100], *clip) if clip else logits[:100]
    diff = np.abs(ref - twin(one_hot_flat(calib[:100], V)).numpy()).max()
    print(f"twin vs original, max |difference| of logits: {diff:.2e}  (must be ~1e-4 or smaller)")

    def representative_dataset():
        for i in range(len(calib)):
            yield [one_hot_flat(calib[i:i + 1], V)]

    conv = tf.lite.TFLiteConverter.from_keras_model(twin)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    with open(a.out, "wb") as f:
        f.write(blob)
    print(f"wrote {a.out}: {len(blob) / 2**20:.2f} MiB")

    interp = tf.lite.Interpreter(model_path=a.out)
    interp.allocate_tensors()
    print("input  quantization (scale, zero point):", interp.get_input_details()[0]["quantization"])
    print("output quantization (scale, zero point):", interp.get_output_details()[0]["quantization"])

    pairs = load_pairs(os.path.join(a.data, "train.jsonl"), per_article=1)
    step = max(1, len(pairs) // a.check)
    pairs = pairs[::step][:a.check]
    f_ok = i_ok = 0
    for p in pairs:
        f_ok += generate(model, p["q"], stoi, vocab, W) == p["a"]
        i_ok += generate_tflite(interp, p["q"], stoi, vocab, W, V) == p["a"]
    print(f"exact match on {len(pairs)} questions: float {f_ok / len(pairs):.2f}   int8 {i_ok / len(pairs):.2f}")


if __name__ == "__main__":
    main()
