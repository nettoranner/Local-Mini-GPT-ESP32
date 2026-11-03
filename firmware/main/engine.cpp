#include "engine.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "ui.h"

namespace {

const char* TAG = "engine";

struct Question { char text[160]; };

QueueHandle_t g_queue = nullptr;
CmlpModel* g_model = nullptr;
GenConfig g_cfg;

struct SinkState {
  std::string text;
  int64_t last_ui_us = 0;
};

// Called for every generated symbol: collect the text and refresh the screen a few times per second.
bool Sink(const char* utf8, void* user) {
  SinkState* s = static_cast<SinkState*>(user);
  s->text += utf8;
  const int64_t now = esp_timer_get_time();
  if (now - s->last_ui_us >= 80 * 1000) {
    ui_set_answer(s->text.c_str());
    s->last_ui_us = now;
  }
  vTaskDelay(1);   // lets the idle task run (task watchdog) and the UI breathe
  return true;
}

void EngineTask(void*) {
  for (;;) {
    Question q;
    xQueueReceive(g_queue, &q, portMAX_DELAY);
    ui_set_busy(true);
    ui_set_status("Thinking...");

    SinkState state;
    const GenStats st = GenerateAnswer(*g_model, q.text, g_cfg, Sink, &state);

    ui_set_answer(state.text.c_str());
    char status[64];
    if (st.chars > 0)
      snprintf(status, sizeof status, "%d chars, %.1f ms/char%s", st.chars,
               st.infer_us / 1000.0 / st.chars, st.ended ? "" : " (cut)");
    else
      snprintf(status, sizeof status, "no answer");
    ui_set_status(status);
    ui_set_busy(false);
    ESP_LOGI(TAG, "Q: \"%s\" -> %d chars, %.2f ms/char", q.text, st.chars,
             st.chars ? st.infer_us / 1000.0 / st.chars : 0.0);
  }
}

}  // namespace

bool EngineStart(CmlpModel* model, const GenConfig& cfg) {
  g_model = model;
  g_cfg = cfg;
  g_queue = xQueueCreate(1, sizeof(Question));
  if (!g_queue) return false;
  return xTaskCreatePinnedToCore(EngineTask, "engine", 12288, nullptr, 3, nullptr, 1) == pdPASS;
}

bool EngineAsk(const char* question) {
  if (!g_queue || !question) return false;
  Question q;
  strncpy(q.text, question, sizeof(q.text) - 1);
  q.text[sizeof(q.text) - 1] = 0;
  return xQueueSend(g_queue, &q, 0) == pdPASS;
}
