"""dataset.py - turn question/answer pairs into (256 previous symbols -> next symbol) examples.

Each pair becomes the symbol sequence   question + "\\n" + answer + END.
For every symbol of the ANSWER (and END) one training example is made:
    input  = the 256 symbols before it (left-padded with 0 = PAD)
    target = that symbol
Symbols of the question are never targets, they are only context.
"""
import json

import numpy as np

END = "\x03"
PAD = "\x00"


def load_vocab(path):
    vocab = json.load(open(path, encoding="utf-8"))
    if vocab[0] != PAD:
        raise SystemExit('vocab[0] must be the padding symbol "\\x00". Put it first in VOCAB in '
                         "build_qa.py and rebuild the data.")
    return vocab, {ch: i for i, ch in enumerate(vocab)}


def load_pairs(path, limit_articles=None, per_article=None, only_answers=None):
    """Read a .jsonl of {"q","a"}.  limit_articles / per_article shrink the data for quick tests."""
    pairs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if only_answers is not None:
        pairs = [p for p in pairs if p["a"] in only_answers]
    if limit_articles:
        keep = set(list(dict.fromkeys(p["a"] for p in pairs))[:limit_articles])
        pairs = [p for p in pairs if p["a"] in keep]
    if per_article:
        count, out = {}, []
        for p in pairs:
            count[p["a"]] = count.get(p["a"], 0) + 1
            if count[p["a"]] <= per_article:
                out.append(p)
        pairs = out
    return pairs


def encode_pair(q, a, stoi):
    """Return (indices of 'question\\nanswer<END>', index where the answer starts)."""
    qi = [stoi[c] for c in q + "\n"]
    ai = [stoi[c] for c in a] + [stoi[END]]
    return qi + ai, len(qi)


class Examples:
    """All training examples. Windows are cut out batch by batch, so memory stays small
    (1.3 million examples x 256 symbols would otherwise be ~1 GB)."""

    def __init__(self, pairs, stoi, window=256):
        self.window = window
        self.seqs, self.index = [], []      # index: (sequence number, position of the target)
        for p in pairs:
            seq, start = encode_pair(p["q"], p["a"], stoi)
            self.seqs.append(np.array(seq, dtype=np.int32))
            n = len(self.seqs) - 1
            self.index += [(n, i) for i in range(start, len(seq))]
        self.index = np.array(self.index, dtype=np.int32)

    def __len__(self):
        return len(self.index)

    def window_at(self, n, i):
        w = self.seqs[n][max(0, i - self.window):i]
        return np.concatenate([np.zeros(self.window - len(w), dtype=np.int32), w])

    def batches(self, batch_size, shuffle=True, seed=0):
        order = np.arange(len(self.index))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for s in range(0, len(order), batch_size):
            ids = self.index[order[s:s + batch_size]]
            x = np.stack([self.window_at(n, i) for n, i in ids])
            y = np.array([self.seqs[n][i] for n, i in ids], dtype=np.int32)
            yield x, y


def show_example(ex, vocab, k=0):
    """Print example k back as text ('_' = padding). Use it to check the windows by eye."""
    n, i = ex.index[k]
    x = ex.window_at(n, i)
    print("window:", repr("".join(vocab[t] if t else "_" for t in x)))
    print("target:", repr(vocab[ex.seqs[n][i]]))
