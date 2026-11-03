"""Shared helpers for the CMLP model format (the C++ side is main/cmlp_model.h).

Network:  one-hot(ctx_len x vocab_in)  ->  Dense(hidden) + ReLU  ->  Dense(vocab_out)  ->  float logits

File layout (little endian):
    [0, 128)   header (see _HDR below, padded with zeros)
    then       w1t     int8  [ctx_len*vocab_in][hidden]   first-layer table, row p*vocab_in+symbol
               w2      int8  [vocab_out][hidden]
               b1      int32 [hidden]
               qmul1   int32 [hidden]   requantisation multiplier of layer 1
               qshift1 int32 [hidden]   requantisation shift of layer 1
               b2      int32 [vocab_out]
               scale2  float32 [vocab_out]   logit = (int32 accumulator + b2) * scale2
    every block is padded to a multiple of 16 bytes; the header stores the offsets and a CRC32 of
    everything after the header.
"""
import math
import struct
import zlib

import numpy as np

MAGIC = b"CMLP"
VERSION = 1
HEADER_SIZE = 128
# magic, version, ctx_len, vocab_in, hidden, vocab_out | in_delta, h_zp, act_min, act_max |
# off_w1t, off_w2, off_b1, off_qmul1, off_qshift1, off_b2, off_scale2 | total_size, payload_crc32
_HDR = struct.Struct("<4s5I4i7I2I")
assert _HDR.size == 76
BLOCKS = ["w1t", "w2", "b1", "qmul1", "qshift1", "b2", "scale2"]


class CmlpModel:
    """Plain container for everything the firmware needs."""

    def __init__(self, ctx_len, vocab_in, hidden, vocab_out, in_delta, h_zp, act_min, act_max,
                 w1t, w2, b1, qmul1, qshift1, b2, scale2):
        self.ctx_len, self.vocab_in, self.hidden, self.vocab_out = ctx_len, vocab_in, hidden, vocab_out
        self.in_delta, self.h_zp, self.act_min, self.act_max = in_delta, h_zp, act_min, act_max
        self.w1t = np.ascontiguousarray(w1t, dtype=np.int8)
        self.w2 = np.ascontiguousarray(w2, dtype=np.int8)
        self.b1 = np.ascontiguousarray(b1, dtype="<i4")
        self.qmul1 = np.ascontiguousarray(qmul1, dtype="<i4")
        self.qshift1 = np.ascontiguousarray(qshift1, dtype="<i4")
        self.b2 = np.ascontiguousarray(b2, dtype="<i4")
        self.scale2 = np.ascontiguousarray(scale2, dtype="<f4")
        assert self.w1t.shape == (ctx_len * vocab_in, hidden)
        assert self.w2.shape == (vocab_out, hidden)


def _align(n, a=16):
    return (n + a - 1) // a * a


def pack(m):
    """Serialise a CmlpModel to bytes."""
    payload = bytearray()
    offsets = {}
    pos = HEADER_SIZE
    for name in BLOCKS:
        raw = getattr(m, name).tobytes()
        offsets[name] = pos
        pad = _align(len(raw)) - len(raw)
        payload += raw + b"\0" * pad
        pos += len(raw) + pad
    total = HEADER_SIZE + len(payload)
    crc = zlib.crc32(bytes(payload)) & 0xFFFFFFFF
    head = _HDR.pack(MAGIC, VERSION, m.ctx_len, m.vocab_in, m.hidden, m.vocab_out,
                     m.in_delta, m.h_zp, m.act_min, m.act_max,
                     *[offsets[n] for n in BLOCKS], total, crc)
    return head + b"\0" * (HEADER_SIZE - len(head)) + bytes(payload)


def load(path):
    """Read a .bin written by pack(); also checks the CRC. Returns (CmlpModel, payload_crc32)."""
    blob = open(path, "rb").read()
    f = _HDR.unpack_from(blob, 0)
    magic, version, ctx, vin, hid, vout = f[:6]
    in_delta, h_zp, act_min, act_max = f[6:10]
    offs = dict(zip(BLOCKS, f[10:17]))
    total, crc = f[17], f[18]
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a CMLP v1 file")
    if total != len(blob):
        raise ValueError(f"size mismatch: header says {total}, file has {len(blob)}")
    if zlib.crc32(blob[HEADER_SIZE:]) & 0xFFFFFFFF != crc:
        raise ValueError("CRC mismatch")

    def arr(name, dtype, count):
        return np.frombuffer(blob, dtype=dtype, count=count, offset=offs[name]).copy()

    m = CmlpModel(ctx, vin, hid, vout, in_delta, h_zp, act_min, act_max,
                  arr("w1t", np.int8, ctx * vin * hid).reshape(ctx * vin, hid),
                  arr("w2", np.int8, vout * hid).reshape(vout, hid),
                  arr("b1", "<i4", hid), arr("qmul1", "<i4", hid), arr("qshift1", "<i4", hid),
                  arr("b2", "<i4", vout), arr("scale2", "<f4", vout))
    return m, crc


def quantize_multiplier(mult):
    """Same as tflite::QuantizeMultiplier: mult ~ qm * 2^(shift-31)."""
    if mult == 0.0:
        return 0, 0
    q, s = math.frexp(mult)
    qf = int(math.floor(q * (1 << 31) + 0.5))
    if qf == (1 << 31):
        qf //= 2
        s += 1
    if s < -31:
        return 0, 0
    return qf, s


def mul_by_quant_mul(x, qm, shift):
    """Same as the C++ MulByQuantMul (single rounding). All arguments are int64 arrays."""
    total = 31 - shift
    rnd = np.left_shift(np.int64(1), total - 1)
    return (x * qm + rnd) >> total


def predict(m, window):
    """Bit-exact Python twin of CmlpModel::Predict. window: ctx_len symbol indices.
    Returns (hidden int8 [hidden], logits float32 [vocab_out])."""
    window = np.asarray(window, dtype=np.int64)
    rows = m.w1t[np.arange(m.ctx_len, dtype=np.int64) * m.vocab_in + window]   # [ctx_len, hidden]
    acc = rows.astype(np.int32).sum(axis=0, dtype=np.int32)
    x = acc.astype(np.int64) * m.in_delta + m.b1.astype(np.int64)
    v = mul_by_quant_mul(x, m.qmul1.astype(np.int64), m.qshift1.astype(np.int64)) + m.h_zp
    h = np.clip(v, m.act_min, m.act_max).astype(np.int8)
    s = (h.astype(np.int32) - np.int32(m.h_zp)) @ m.w2.astype(np.int32).T
    logits = (s.astype(np.int32) + m.b2).astype(np.float32) * m.scale2
    return h, logits
