// Boot self-test: runs the model on reference windows computed on the PC (golden_data.h)
// and compares the hidden layer and the logits.
#pragma once
#include <cstddef>

#include "cmlp_model.h"

enum SelfTestResult { kSelfTestSkipped, kSelfTestPass, kSelfTestFail };

// summary receives a one-line description of the outcome.
SelfTestResult RunSelfTest(CmlpModel& model, char* summary, size_t summary_len);
