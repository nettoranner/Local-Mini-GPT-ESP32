#include "cmlp_model.h"

#include <cstdio>
#include <cstring>

#include "platform_mem.h"

namespace {

// Same as tflite::MultiplyByQuantizedMultiplier with single rounding (the Python twin does the same).
inline int32_t MulByQuantMul(int32_t x, int32_t qm, int32_t shift) {
  const int total = 31 - shift;
  const int64_t round = int64_t(1) << (total - 1);
  return static_cast<int32_t>((int64_t(x) * qm + round) >> total);
}

uint32_t g_crc_table[256];
bool g_crc_ready = false;

void InitCrcTable() {
  if (g_crc_ready) return;
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    g_crc_table[i] = c;
  }
  g_crc_ready = true;
}

}  // namespace

CmlpModel::~CmlpModel() {
  FreeMem(w2_); FreeMem(b1_); FreeMem(qmul1_); FreeMem(qshift1_);
  FreeMem(b2_); FreeMem(scale2_); FreeMem(acc_); FreeMem(hidden_);
}

bool CmlpModel::Init(const uint8_t* data, size_t avail, char* err, size_t err_len) {
#define FAIL(...) do { snprintf(err, err_len, __VA_ARGS__); return false; } while (0)
  if (!data || avail < sizeof(CmlpHeader)) FAIL("model partition is empty");
  memcpy(&hdr_, data, sizeof(hdr_));
  if (memcmp(hdr_.magic, "CMLP", 4) != 0) FAIL("no CMLP model in the partition. Flash model/model.bin");
  if (hdr_.version != 1) FAIL("unsupported model version %u", (unsigned)hdr_.version);

  const size_t L = hdr_.ctx_len, V = hdr_.vocab_in, H = hdr_.hidden, V2 = hdr_.vocab_out;
  if (L < 1 || L > (size_t)kMaxCtx || V < 2 || V > (size_t)kMaxVocab || V2 < 2 || V2 > (size_t)kMaxVocab ||
      H < 1 || H > (size_t)kMaxHidden)
    FAIL("unsupported model size (ctx %u, vocab %u/%u, hidden %u)", (unsigned)L, (unsigned)V, (unsigned)V2,
         (unsigned)H);
  if (hdr_.total_size > avail)
    FAIL("model (%u bytes) does not fit the partition (%u bytes)", (unsigned)hdr_.total_size, (unsigned)avail);

  struct Block { uint32_t off; size_t size; const char* name; };
  const Block blocks[] = {
      {hdr_.off_w1t, L * V * H, "w1t"},          {hdr_.off_w2, V2 * H, "w2"},
      {hdr_.off_b1, 4 * H, "b1"},                {hdr_.off_qmul1, 4 * H, "qmul1"},
      {hdr_.off_qshift1, 4 * H, "qshift1"},      {hdr_.off_b2, 4 * V2, "b2"},
      {hdr_.off_scale2, 4 * V2, "scale2"},
  };
  for (const Block& b : blocks) {
    if (b.off < sizeof(CmlpHeader) || (b.off % 16) != 0 || (uint64_t)b.off + b.size > hdr_.total_size)
      FAIL("bad layout of block '%s'", b.name);
  }

  data_ = data;
  w1t_ = reinterpret_cast<const int8_t*>(data + hdr_.off_w1t);

  w2_ = static_cast<int8_t*>(AllocFast(V2 * H));
  b1_ = static_cast<int32_t*>(AllocFast(4 * H));
  qmul1_ = static_cast<int32_t*>(AllocFast(4 * H));
  qshift1_ = static_cast<int32_t*>(AllocFast(4 * H));
  b2_ = static_cast<int32_t*>(AllocFast(4 * V2));
  scale2_ = static_cast<float*>(AllocFast(4 * V2));
  acc_ = static_cast<int32_t*>(AllocFast(4 * H));
  hidden_ = static_cast<int8_t*>(AllocFast(H));
  if (!w2_ || !b1_ || !qmul1_ || !qshift1_ || !b2_ || !scale2_ || !acc_ || !hidden_) FAIL("out of memory");

  memcpy(w2_, data + hdr_.off_w2, V2 * H);
  memcpy(b1_, data + hdr_.off_b1, 4 * H);
  memcpy(qmul1_, data + hdr_.off_qmul1, 4 * H);
  memcpy(qshift1_, data + hdr_.off_qshift1, 4 * H);
  memcpy(b2_, data + hdr_.off_b2, 4 * V2);
  memcpy(scale2_, data + hdr_.off_scale2, 4 * V2);
  for (size_t o = 0; o < H; ++o)
    if (qshift1_[o] < -31 || qshift1_[o] > 30) FAIL("requantisation shift out of range");
  return true;
#undef FAIL
}

bool CmlpModel::VerifyCrc(void (*progress)(size_t, size_t, void*), void* user) const {
  InitCrcTable();
  uint32_t c = 0xFFFFFFFFu;
  const size_t start = sizeof(CmlpHeader), total = hdr_.total_size;
  const size_t chunk = 256 * 1024;
  for (size_t pos = start; pos < total; pos += chunk) {
    const size_t end = pos + chunk < total ? pos + chunk : total;
    for (size_t i = pos; i < end; ++i) c = g_crc_table[(c ^ data_[i]) & 0xFF] ^ (c >> 8);
    if (progress) progress(end - start, total - start, user);
  }
  return ~c == hdr_.payload_crc32;
}

bool CmlpModel::Predict(const uint16_t* window, float* logits) {
  const int L = ctx_len(), V = vocab_in(), H = hidden(), V2 = vocab_out();

  // Layer 1: sum of table rows for the symbols in the window (rows are read from flash).
  memset(acc_, 0, sizeof(int32_t) * H);
  for (int p = 0; p < L; ++p) {
    const unsigned c = window[p];
    if (c >= (unsigned)V) return false;
    const int8_t* row = w1t_ + (size_t)(p * V + c) * H;
    for (int o = 0; o < H; ++o) acc_[o] += row[o];
  }
  for (int o = 0; o < H; ++o) {
    int32_t v = MulByQuantMul(acc_[o] * hdr_.in_delta + b1_[o], qmul1_[o], qshift1_[o]) + hdr_.h_zp;
    v = v < hdr_.act_min ? hdr_.act_min : (v > hdr_.act_max ? hdr_.act_max : v);
    hidden_[o] = static_cast<int8_t>(v);
  }

  // Layer 2: int32 accumulator -> float logits (no coarse int8 output quantisation).
  for (int o = 0; o < V2; ++o) {
    const int8_t* w = w2_ + (size_t)o * H;
    int32_t s = 0;
    for (int k = 0; k < H; ++k) s += (int32_t(hidden_[k]) - hdr_.h_zp) * w[k];
    logits[o] = static_cast<float>(s + b2_[o]) * scale2_[o];
  }
  return true;
}
