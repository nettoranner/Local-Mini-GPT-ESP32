#pragma once
#include <cstdint>

struct SamplerConfig {
  float temperature = 0.8f;  // <= 0: greedy (argmax)
  int top_k = 0;             // 0: no limit
};

// xorshift32, the state must not be 0
uint32_t RngNext(uint32_t* state);

// scores: the distribution is softmax(scores); -INFINITY is allowed.
// scratch: n floats of work space. Returns the chosen symbol index.
int SampleIndex(const float* scores, int n, const SamplerConfig& cfg, uint32_t* rng, float* scratch);

int ArgMax(const float* v, int n);
