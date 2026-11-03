#include "generate.h"

#include <chrono>
#include <cmath>
#include <cstring>

#include "sampler.h"
#include "vocab.h"

GenStats GenerateAnswer(CmlpModel& model, const char* question, const GenConfig& cfg, CharSink sink,
                        void* user) {
  GenStats st;
  const int L = model.ctx_len();
  const int V = model.vocab_out();

  int space = VocabFind(" ");
  if (space < 0) space = 0;
  const int pad = kVocabPadIndex >= 0 ? kVocabPadIndex : space;

  // Seed = cleaned question + "\n". Control characters become spaces, outer spaces are trimmed.
  char seed[256];
  size_t n = 0;
  const char* p = question ? question : "";
  while (*p == ' ') ++p;
  for (; *p && n < sizeof(seed) - 2; ++p) {
    const unsigned char ch = static_cast<unsigned char>(*p);
    seed[n++] = ch < 32 ? ' ' : static_cast<char>(ch);
  }
  while (n > 0 && seed[n - 1] == ' ') --n;
  seed[n++] = '\n';
  seed[n] = 0;

  uint16_t window[CmlpModel::kMaxCtx];
  for (int i = 0; i < L; ++i) window[i] = static_cast<uint16_t>(pad);
  uint16_t enc[CmlpModel::kMaxCtx];
  int unknown = 0;
  const int k = VocabEncode(seed, enc, L, space, &unknown);   // keeps the last L symbols
  memcpy(window + (L - k), enc, sizeof(uint16_t) * k);

  float scores[CmlpModel::kMaxVocab], scratch[CmlpModel::kMaxVocab];
  SamplerConfig sc;
  sc.temperature = cfg.temperature;
  sc.top_k = cfg.top_k;
  uint32_t rng = 0x2545F491u;

  for (int i = 0; i < cfg.max_chars; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = model.Predict(window, scores);
    st.infer_us += std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t0).count();
    if (!ok) break;

    if (kVocabPadIndex >= 0) scores[kVocabPadIndex] = -INFINITY;   // never emit the padding symbol
    const int c = SampleIndex(scores, V, sc, &rng, scratch);
    if (kVocabEndIndex >= 0 && c == kVocabEndIndex) { st.ended = true; break; }

    const char* text = (c >= 0 && c < kVocabSize) ? kVocab[c] : "";
    if (text[0] && !sink(text, user)) break;
    ++st.chars;

    if (c < model.vocab_in()) {                                    // slide the window
      memmove(window, window + 1, sizeof(uint16_t) * (L - 1));
      window[L - 1] = static_cast<uint16_t>(c);
    }
  }
  return st;
}
