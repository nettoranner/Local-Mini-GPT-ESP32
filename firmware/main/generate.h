// Answer generation: question -> text, one symbol at a time (pure logic, no UI / RTOS).
#pragma once
#include <cstdint>

#include "cmlp_model.h"

struct GenConfig {
  int max_chars = 400;      // hard limit if the model never emits END
  float temperature = 0.f;  // <= 0: always pick the most likely symbol (greedy, repeatable)
  int top_k = 0;            // only used when temperature > 0
};

struct GenStats {
  int chars = 0;            // symbols generated (not counting END)
  bool ended = false;       // true if the model emitted END
  int64_t infer_us = 0;     // time spent in Predict
};

// Called for every generated symbol (UTF-8 text). Return false to stop.
typedef bool (*CharSink)(const char* utf8, void* user);

// The model was trained on  "<question>\n<answer><END>"  so the window is filled with the
// question + "\n" and the answer is then generated until END.
GenStats GenerateAnswer(CmlpModel& model, const char* question, const GenConfig& cfg, CharSink sink,
                        void* user);
