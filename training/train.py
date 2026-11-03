"""train.py - char MLP that memorises  question -> answer  pairs.

Examples (run from the project root):
  # 1) quick check that everything works: 30 topics, few questions each
  python training/train.py --data training/data/qa500 --limit 30 --per-article 6 --epochs 15 --out training/runs/sanity
  # 2) the real run
  python training/train.py --data training/data/qa500 --epochs 20 --out training/runs/qa500
  # continue after a crash / stop
  python training/train.py --data training/data/qa500 --epochs 20 --out training/runs/qa500 --resume
"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import END, Examples, load_pairs, load_vocab, show_example  # noqa: E402

keras = tf.keras


class PositionalOneHotDense(keras.layers.Layer):
    """Exactly Dense(units) applied to a one-hot input of shape (window, vocab), but computed as
    a table lookup: for every position take row (position*vocab + symbol) and sum the rows.
    Same result, ~100x less work. The table (window*vocab, units) IS the Dense kernel, so it can be
    copied into an ordinary Dense layer later (for the .tflite export)."""

    def __init__(self, window, vocab, units, **kw):
        super().__init__(**kw)
        self.window, self.vocab, self.units = window, vocab, units

    def build(self, input_shape):
        self.table = self.add_weight(name="table", shape=(self.window * self.vocab, self.units),
                                     initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros")

    def call(self, x):                                    # x: (batch, window) symbol indices
        offsets = tf.range(self.window, dtype=tf.int32)[None, :] * self.vocab
        rows = tf.gather(self.table, tf.cast(x, tf.int32) + offsets)   # (batch, window, units)
        return tf.reduce_sum(rows, axis=1) + self.bias


def build_model(window, vocab, hidden):
    inp = keras.Input(shape=(window,), dtype="int32")
    h = PositionalOneHotDense(window, vocab, hidden)(inp)
    h = keras.layers.ReLU()(h)
    out = keras.layers.Dense(vocab)(h)                    # logits, softmax is applied in the loss
    return keras.Model(inp, out)


def generate(model, question, stoi, vocab, window, max_len=400):
    """Greedy decoding: always take the most likely next symbol, stop at END."""
    space = stoi[" "]
    seq = [stoi.get(c, space) for c in question + "\n"]
    out = []
    for _ in range(max_len):
        w = seq[-window:]
        x = np.array([[0] * (window - len(w)) + w], dtype=np.int32)
        nxt = int(np.argmax(model(x, training=False).numpy()[0]))
        if vocab[nxt] == END:
            break
        out.append(vocab[nxt])
        seq.append(nxt)
    return "".join(out)


def spread(pairs, k):
    """k pairs spread evenly over the whole list (so all topics are sampled)."""
    step = max(1, len(pairs) // k)
    return pairs[::step][:k]


def exact_match(model, pairs, stoi, vocab, window):
    """Share of questions for which the generated text equals the stored answer exactly."""
    if not pairs:
        return float("nan")
    hit = sum(generate(model, p["q"], stoi, vocab, window) == p["a"] for p in pairs)
    return hit / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train.jsonl, val.jsonl, vocab.json")
    ap.add_argument("--out", required=True, help="folder for checkpoints")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=480)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N topics (quick check)")
    ap.add_argument("--per-article", type=int, default=None, help="use at most K questions per topic")
    ap.add_argument("--eval-every", type=int, default=5, help="epochs between exact-match checks")
    ap.add_argument("--eval-pairs", type=int, default=40, help="how many questions to generate when checking")
    ap.add_argument("--ask", default="who is Nikola Tesla?", help="question printed at the end")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vocab, stoi = load_vocab(os.path.join(a.data, "vocab.json"))
    train_pairs = load_pairs(os.path.join(a.data, "train.jsonl"), a.limit, a.per_article)
    answers = {p["a"] for p in train_pairs}
    val_path = os.path.join(a.data, "val.jsonl")
    val_pairs = load_pairs(val_path, only_answers=answers) if os.path.exists(val_path) else []
    train = Examples(train_pairs, stoi, a.window)
    print(f"vocab {len(vocab)}, window {a.window}, topics {len(answers)}, "
          f"train pairs {len(train_pairs)}, train examples {len(train):,}")
    show_example(train, vocab, 0)

    model = build_model(a.window, len(vocab), a.hidden)
    model.summary()
    ckpt = os.path.join(a.out, "last.weights.h5")
    if a.resume and os.path.exists(ckpt):
        model.load_weights(ckpt)
        print("resumed from", ckpt)

    steps_per_epoch = -(-len(train) // a.batch)
    schedule = keras.optimizers.schedules.CosineDecay(a.lr, a.epochs * steps_per_epoch, alpha=0.01)
    opt = keras.optimizers.Adam(schedule)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.convert_to_tensor(g) for g in grads] # gather gives sparse grads -> make dense
        opt.apply_gradients(zip(grads, model.trainable_variables))
        acc = tf.reduce_mean(tf.cast(tf.argmax(logits, -1, output_type=tf.int32) == y, tf.float32))
        return loss, acc

    for ep in range(1, a.epochs + 1):
        t0, losses, accs = time.time(), [], []
        for x, y in train.batches(a.batch, seed=ep):
            l, ac = step(x, y)
            losses.append(float(l))
            accs.append(float(ac))
        model.save_weights(ckpt) # a crash never costs more than one epoch
        line = (f"epoch {ep}/{a.epochs}  loss {np.mean(losses):.4f}  "
                f"next-char acc {np.mean(accs):.3f}  {time.time() - t0:.0f}s")
        if ep % a.eval_every == 0 or ep == a.epochs:
            em_tr = exact_match(model, train_pairs[:a.eval_pairs], stoi, vocab, a.window)
            line += f"  exact(train questions) {em_tr:.2f}"
            if val_pairs:
                em_va = exact_match(model, spread(val_pairs, a.eval_pairs), stoi, vocab, a.window)
                line += f"  exact(NEW wording) {em_va:.2f}"
        print(line, flush=True)

    print()
    print(a.ask, "->", generate(model, a.ask, stoi, vocab, a.window))


if __name__ == "__main__":
    main()
