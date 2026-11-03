// Background task that answers questions, so the touch UI never freezes.
#pragma once
#include "cmlp_model.h"
#include "generate.h"

// Starts the task (pinned to core 1). The model must outlive it.
bool EngineStart(CmlpModel* model, const GenConfig& cfg);

// Queues a question (copied). Returns false if the engine is busy or not started. Never blocks.
bool EngineAsk(const char* question);
