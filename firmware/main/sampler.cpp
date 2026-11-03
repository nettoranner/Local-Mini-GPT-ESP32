#include "sampler.h"

#include <cmath>

uint32_t RngNext(uint32_t* s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return *s = x;
}

int ArgMax(const float* v, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

int SampleIndex(const float* scores, int n, const SamplerConfig& cfg, uint32_t* rng, float* p) {
  const int best = ArgMax(scores, n);
  if (cfg.temperature <= 0.f || std::isinf(scores[best])) return best;

  // top-k threshold: the k-th largest value (n is small, a plain scan is enough).
  float thr = -INFINITY;
  if (cfg.top_k > 0 && cfg.top_k < n) {
    for (int i = 0; i < n; ++i) {
      int greater = 0;
      for (int j = 0; j < n; ++j) greater += scores[j] > scores[i];
      if (greater < cfg.top_k && scores[i] > -INFINITY && (thr == -INFINITY || scores[i] < thr))
        thr = scores[i];
    }
  }

  const float inv_t = 1.f / cfg.temperature;
  const float mx = scores[best];
  float sum = 0.f;
  for (int i = 0; i < n; ++i) {
    const float s = scores[i];
    p[i] = (s == -INFINITY || s < thr) ? 0.f : expf((s - mx) * inv_t);
    sum += p[i];
  }
  const float r = (RngNext(rng) >> 8) * (1.f / 16777216.f) * sum;
  float acc = 0.f;
  for (int i = 0; i < n; ++i) {
    acc += p[i];
    if (r < acc) return i;
  }
  return best;
}
