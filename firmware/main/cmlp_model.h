// The "CMLP" model: a small int8 MLP that lives in a flash partition and is read in place.
//
//   one-hot(ctx_len x vocab_in) -> Dense(hidden) + ReLU -> Dense(vocab_out) -> float logits
//
// Because the input is one-hot, the first layer is just the SUM of ctx_len rows of a table:
// for window position p holding symbol c, take row (p * vocab_in + c) of W1^T (hidden int8 values).
// W1^T is stored row-major in the partition, so every row is contiguous in flash and is fetched through
// the flash cache (memory-mapped partition). Nothing big is copied to RAM; only ~60 KB of hot arrays are.
// The format is written by tools/export_model_bin.py; tools/cmlp.py is its bit-exact Python twin.
#pragma once
#include <cstddef>
#include <cstdint>

struct CmlpHeader {
  char magic[4];                  // "CMLP"
  uint32_t version;               // 1
  uint32_t ctx_len, vocab_in, hidden, vocab_out;
  int32_t in_delta;               // quantised 1.0 minus zero point of the input
  int32_t h_zp, act_min, act_max; // hidden layer: zero point and ReLU clamp
  uint32_t off_w1t, off_w2, off_b1, off_qmul1, off_qshift1, off_b2, off_scale2;
  uint32_t total_size;            // whole file, bytes
  uint32_t payload_crc32;         // CRC32 of everything after the header
  uint8_t reserved[128 - 76];
};
static_assert(sizeof(CmlpHeader) == 128, "CmlpHeader must be 128 bytes");

class CmlpModel {
 public:
  static constexpr int kMaxCtx = 512, kMaxVocab = 512, kMaxHidden = 4096;

  ~CmlpModel();
  // data: start of the mapped partition, avail: its size. On failure writes a message to err.
  bool Init(const uint8_t* data, size_t avail, char* err, size_t err_len);
  // Re-reads the whole payload and compares the CRC (a few hundred ms for 12 MB).
  // progress is called every 256 KB so the caller can update a progress display / yield.
  bool VerifyCrc(void (*progress)(size_t done, size_t total, void* user), void* user) const;
  // window: ctx_len symbol indices in [0, vocab_in), oldest first. logits: vocab_out floats.
  bool Predict(const uint16_t* window, float* logits);

  int ctx_len() const { return static_cast<int>(hdr_.ctx_len); }
  int vocab_in() const { return static_cast<int>(hdr_.vocab_in); }
  int vocab_out() const { return static_cast<int>(hdr_.vocab_out); }
  int hidden() const { return static_cast<int>(hdr_.hidden); }
  uint32_t payload_crc() const { return hdr_.payload_crc32; }
  const int8_t* Hidden() const { return hidden_; }  // hidden layer after the last Predict (self-test)

 private:
  CmlpHeader hdr_{};
  const uint8_t* data_ = nullptr;
  const int8_t* w1t_ = nullptr;  // stays in flash
  // hot data, copied to internal RAM:
  int8_t* w2_ = nullptr;         // [vocab_out][hidden]
  int32_t* b1_ = nullptr;        // [hidden]
  int32_t* qmul1_ = nullptr;
  int32_t* qshift1_ = nullptr;
  int32_t* b2_ = nullptr;        // [vocab_out]
  float* scale2_ = nullptr;      // [vocab_out]
  int32_t* acc_ = nullptr;       // [hidden] work buffer
  int8_t* hidden_ = nullptr;     // [hidden]
};
