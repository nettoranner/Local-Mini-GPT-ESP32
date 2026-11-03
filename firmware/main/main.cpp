// WikiGPT on ESP32-S3: a 12-million-parameter int8 character model that answers questions offline.
// Touch screen in, touch screen out. The model lives in the flash partition "model" and is read in place.
//
// Boot sequence: display + touch -> (first boot: touch calibration) -> map model -> check CRC ->
// self-test -> start the answer task -> main screen.

#include <cstdio>
#include <cstring>

#include "board.h"
#include "cmlp_model.h"
#include "engine.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "self_test.h"
#include "touch_calib.h"
#include "ui.h"

static const char* TAG = "main";

static CmlpModel g_model;

[[noreturn]] static void Fail(const char* what) {
  ESP_LOGE(TAG, "%s", what);
  ui_show_message("Error", what);
  for (;;) vTaskDelay(portMAX_DELAY);
}

static bool OnAsk(const char* question) { return EngineAsk(question); }

static void CrcProgress(size_t done, size_t total, void*) {
  char buf[48];
  snprintf(buf, sizeof buf, "Checking model... %d%%", (int)(done * 100 / total));
  ui_show_message("WikiGPT", buf);
  vTaskDelay(1);
}

static const uint8_t* MapModel(size_t* size) {
  const esp_partition_t* part =
      esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
  if (!part) return nullptr;
  const void* ptr = nullptr;
  esp_partition_mmap_handle_t handle;
  if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &ptr, &handle) != ESP_OK)
    return nullptr;
  *size = part->size;
  ESP_LOGI(TAG, "model partition at 0x%lx, %lu KB, mapped at %p", (unsigned long)part->address,
           (unsigned long)(part->size / 1024), ptr);
  return static_cast<const uint8_t*>(ptr);
}

extern "C" void app_main(void) {
  esp_err_t r = nvs_flash_init();
  if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }

  lv_display_t* disp = board_init();
  if (!disp) {
    ESP_LOGE(TAG, "display init failed (see log above)");
    for (;;) vTaskDelay(portMAX_DELAY);
  }
  ui_init(disp, OnAsk);
  ui_show_message("WikiGPT", "Starting...");

  touch_calib_load();
  if (!touch_calib_valid()) ui_run_calibration();

  // ---- model ----
  ui_show_message("WikiGPT", "Loading model...");
  size_t part_size = 0;
  const uint8_t* data = MapModel(&part_size);
  if (!data) Fail("Partition 'model' not found or cannot be mapped. Check partitions.csv.");

  char err[160];
  if (!g_model.Init(data, part_size, err, sizeof err)) Fail(err);
  ESP_LOGI(TAG, "model: window %d, vocab %d/%d, hidden %d", g_model.ctx_len(), g_model.vocab_in(),
           g_model.vocab_out(), g_model.hidden());

  if (!g_model.VerifyCrc(CrcProgress, nullptr))
    Fail("Model CRC mismatch. Flash the model again: idf.py flash");

  char summary[160];
  const SelfTestResult st = RunSelfTest(g_model, summary, sizeof summary);
  ESP_LOGI(TAG, "%s", summary);
  if (st == kSelfTestFail) Fail(summary);

  GenConfig cfg;
  if (!EngineStart(&g_model, cfg)) Fail("Could not start the answer task");

  ui_show_main();
  ui_set_status(st == kSelfTestPass ? "Ready (self-test OK)" : "Ready (no self-test)");

  for (;;) {
    if (ui_recalibrate_requested()) {
      ui_run_calibration();
      ui_show_main();
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}
