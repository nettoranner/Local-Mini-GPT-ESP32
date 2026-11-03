#include "touch_calib.h"

#include <math.h>
#include <stdlib.h>

#include "board_config.h"
#include "esp_log.h"
#include "nvs.h"

static const char* TAG = "touch_calib";

typedef struct {
  uint32_t version;
  uint32_t swap;     // 1: screen X follows raw Y and screen Y follows raw X
  float ax, bx;      // screen_x = ax * u + bx,  u = swap ? raw_y : raw_x
  float ay, by;      // screen_y = ay * v + by,  v = swap ? raw_x : raw_y
} calib_blob_t;

static calib_blob_t s_c;
static bool s_valid = false;

bool touch_calib_valid(void) { return s_valid; }

void touch_calib_invalidate(void) { s_valid = false; }

bool touch_calib_load(void) {
  nvs_handle_t h;
  if (nvs_open("wikigpt", NVS_READONLY, &h) != ESP_OK) return false;
  calib_blob_t c;
  size_t len = sizeof(c);
  const esp_err_t err = nvs_get_blob(h, "touch", &c, &len);
  nvs_close(h);
  if (err != ESP_OK || len != sizeof(c) || c.version != 1) return false;
  s_c = c;
  s_valid = true;
  ESP_LOGI(TAG, "loaded: swap=%u ax=%.4f bx=%.1f ay=%.4f by=%.1f", (unsigned)c.swap, c.ax, c.bx, c.ay, c.by);
  return true;
}

bool touch_calib_save(void) {
  nvs_handle_t h;
  if (nvs_open("wikigpt", NVS_READWRITE, &h) != ESP_OK) return false;
  esp_err_t err = nvs_set_blob(h, "touch", &s_c, sizeof(s_c));
  if (err == ESP_OK) err = nvs_commit(h);
  nvs_close(h);
  return err == ESP_OK;
}

bool touch_calib_compute(const uint16_t raw1[2], const uint16_t raw2[2], const uint16_t raw3[2]) {
  const int x1 = CAL_MARGIN, y1 = CAL_MARGIN;
  const int x2 = UI_WIDTH - 1 - CAL_MARGIN;
  const int y3 = UI_HEIGHT - 1 - CAL_MARGIN;

  // P1 -> P2 moves only along screen X: the raw axis that changes more is the one driving screen X.
  const int dx = abs((int)raw2[0] - (int)raw1[0]);
  const int dy = abs((int)raw2[1] - (int)raw1[1]);
  const bool swap = dy > dx;
  const int u1 = swap ? raw1[1] : raw1[0], u2 = swap ? raw2[1] : raw2[0];
  const int v1 = swap ? raw1[0] : raw1[1], v3 = swap ? raw3[0] : raw3[1];
  const int du = u2 - u1, dv = v3 - v1;
  ESP_LOGI(TAG, "raw1=(%u,%u) raw2=(%u,%u) raw3=(%u,%u) swap=%d du=%d dv=%d", raw1[0], raw1[1], raw2[0],
           raw2[1], raw3[0], raw3[1], (int)swap, du, dv);
  if (abs(du) < 400 || abs(dv) < 400) return false;   // taps too close together

  s_c.version = 1;
  s_c.swap = swap ? 1u : 0u;
  s_c.ax = (float)(x2 - x1) / (float)du;
  s_c.bx = (float)x1 - s_c.ax * (float)u1;
  s_c.ay = (float)(y3 - y1) / (float)dv;
  s_c.by = (float)y1 - s_c.ay * (float)v1;
  s_valid = true;
  return true;
}

void touch_calib_apply(uint16_t raw_x, uint16_t raw_y, int* sx, int* sy) {
  float x, y;
  if (s_valid) {
    const float u = s_c.swap ? (float)raw_y : (float)raw_x;
    const float v = s_c.swap ? (float)raw_x : (float)raw_y;
    x = s_c.ax * u + s_c.bx;
    y = s_c.ay * v + s_c.by;
  } else {   // rough default so the calibration screen can still receive taps
    x = (float)raw_x * UI_WIDTH / 4096.0f;
    y = (float)raw_y * UI_HEIGHT / 4096.0f;
  }
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x > UI_WIDTH - 1) x = UI_WIDTH - 1;
  if (y > UI_HEIGHT - 1) y = UI_HEIGHT - 1;
  *sx = (int)(x + 0.5f);
  *sy = (int)(y + 0.5f);
}
