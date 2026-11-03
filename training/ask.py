"""ask.py - talk to a trained model on the PC (same greedy decoding the board will do).

  python training/ask.py --run training/runs/qa500 --data training/data/qa500
Empty line = exit.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_vocab  # noqa: E402
from train import build_model, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with last.weights.h5")
    ap.add_argument("--data", required=True, help="folder with vocab.json")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=480)
    a = ap.parse_args()

    vocab, stoi = load_vocab(os.path.join(a.data, "vocab.json"))
    model = build_model(a.window, len(vocab), a.hidden)
    model.load_weights(os.path.join(a.run, "last.weights.h5"))
    while True:
        q = input("> ").strip()
        if not q:
            break
        print(generate(model, q, stoi, vocab, a.window))


if __name__ == "__main__":
    main()
