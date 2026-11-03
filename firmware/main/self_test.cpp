#include "self_test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "golden_data.h"
#include "sampler.h"

SelfTestResult RunSelfTest(CmlpModel& model, char* summary, size_t n) {
#if GOLDEN_VALID
  if (model.payload_crc() != kGoldenPayloadCrc) {
    snprintf(summary, n, "golden data belongs to another model: run tools/make_golden.py");
    return kSelfTestSkipped;
  }
  if (model.ctx_len() != GOLDEN_CTX_LEN || model.hidden() != GOLDEN_HIDDEN ||
      model.vocab_out() != GOLDEN_OUT_VOCAB) {
    snprintf(summary, n, "golden data has other dimensions: run tools/make_golden.py");
    return kSelfTestSkipped;
  }
  float logits[GOLDEN_OUT_VOCAB];
  int worst_hidden = 0, top1_mismatch = 0;
  float worst_rel = 0.f;
  for (int c = 0; c < GOLDEN_NUM_CASES; ++c) {
    if (!model.Predict(kGoldenInput[c], logits)) {
      snprintf(summary, n, "Predict failed");
      return kSelfTestFail;
    }
    const int8_t* h = model.Hidden();
    for (int i = 0; i < GOLDEN_HIDDEN; ++i) {
      const int d = abs(h[i] - kGoldenHidden[c][i]);
      if (d > worst_hidden) worst_hidden = d;
    }
    for (int i = 0; i < GOLDEN_OUT_VOCAB; ++i) {
      const float g = kGoldenLogits[c][i];
      const float rel = fabsf(logits[i] - g) / (fabsf(g) + 1e-3f);
      if (rel > worst_rel) worst_rel = rel;
    }
    if (ArgMax(logits, GOLDEN_OUT_VOCAB) != ArgMax(kGoldenLogits[c], GOLDEN_OUT_VOCAB)) ++top1_mismatch;
  }
  const bool pass = worst_hidden == 0 && worst_rel < 1e-4f && top1_mismatch == 0;
  snprintf(summary, n, "self-test %s: hidden diff %d, logit rel.err %.1e, top-1 mismatches %d",
           pass ? "OK" : "FAILED", worst_hidden, (double)worst_rel, top1_mismatch);
  return pass ? kSelfTestPass : kSelfTestFail;
#else
  (void)model;
  snprintf(summary, n, "no golden data: run tools/make_golden.py");
  return kSelfTestSkipped;
#endif
}
