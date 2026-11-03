#include "ui.h"

#include <stdio.h>
#include <string.h>

#include "board.h"
#include "board_config.h"
#include "esp_log.h"
#include "esp_lvgl_port.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "touch_calib.h"

static const char* TAG = "ui";

#define COL_BG      0x14181C
#define COL_TEXT    0xE6EAEE
#define COL_DIM     0x8A949E
#define COL_ACCENT  0x2E7DFF

static ui_ask_cb_t s_on_ask;
static volatile bool s_busy;
static volatile bool s_recal;

// splash
static lv_obj_t *s_splash, *s_splash_title, *s_splash_text;
// main screen
static lv_obj_t *s_main, *s_title, *s_status, *s_ta, *s_btn, *s_answer_box, *s_answer, *s_kb;
// calibration overlay
static lv_obj_t *s_cal, *s_cal_dot, *s_cal_label;
static SemaphoreHandle_t s_cal_done;
static int s_cal_step;
static uint16_t s_cal_raw[3][2];

// ------------------------------------------------------------------ helpers
static lv_obj_t* make_plain_container(lv_obj_t* parent, int x, int y, int w, int h) {
  lv_obj_t* c = lv_obj_create(parent);
  lv_obj_set_pos(c, x, y);
  lv_obj_set_size(c, w, h);
  lv_obj_set_style_bg_opa(c, LV_OPA_TRANSP, 0);
  lv_obj_set_style_border_width(c, 0, 0);
  lv_obj_set_style_pad_all(c, 0, 0);
  lv_obj_set_style_radius(c, 0, 0);
  lv_obj_remove_flag(c, LV_OBJ_FLAG_SCROLLABLE);
  return c;
}

static lv_obj_t* make_label(lv_obj_t* parent, const char* text) {
  lv_obj_t* l = lv_label_create(parent);
  lv_label_set_text(l, text);
  lv_obj_set_style_text_color(l, lv_color_hex(COL_TEXT), 0);
  return l;
}

static void set_busy_locked(bool busy) {
  s_busy = busy;
  if (busy) lv_obj_add_state(s_btn, LV_STATE_DISABLED);
  else lv_obj_remove_state(s_btn, LV_STATE_DISABLED);
}

// ------------------------------------------------------------------ main screen events
static void submit(void) {
  if (s_busy) return;
  const char* text = lv_textarea_get_text(s_ta);
  if (!text || !text[0]) return;
  lv_obj_add_flag(s_kb, LV_OBJ_FLAG_HIDDEN);
  lv_label_set_text(s_answer, "");
  if (s_on_ask && s_on_ask(text)) {
    set_busy_locked(true);
    lv_label_set_text(s_status, "Thinking...");
  }
}

static void ta_event_cb(lv_event_t* e) {
  const lv_event_code_t code = lv_event_get_code(e);
  if (code == LV_EVENT_CLICKED || code == LV_EVENT_FOCUSED) {
    lv_keyboard_set_textarea(s_kb, s_ta);
    lv_obj_remove_flag(s_kb, LV_OBJ_FLAG_HIDDEN);
  } else if (code == LV_EVENT_READY) {
    submit();
  }
}

static void kb_event_cb(lv_event_t* e) {
  const lv_event_code_t code = lv_event_get_code(e);
  if (code == LV_EVENT_READY) submit();                           // the "OK" key
  else if (code == LV_EVENT_CANCEL) lv_obj_add_flag(s_kb, LV_OBJ_FLAG_HIDDEN);   // the "hide keyboard" key
}

static void btn_event_cb(lv_event_t* e) {
  (void)e;
  submit();
}

static void title_event_cb(lv_event_t* e) {
  (void)e;
  s_recal = true;   // polled by app_main
}

// ------------------------------------------------------------------ calibration screen
static void cal_show_step_locked(void) {
  static const char* kText[3] = {"Tap the cross at the TOP-LEFT", "Tap the cross at the TOP-RIGHT",
                                 "Tap the cross at the BOTTOM-LEFT"};
  int x = CAL_MARGIN, y = CAL_MARGIN;
  if (s_cal_step == 1) x = UI_WIDTH - 1 - CAL_MARGIN;
  if (s_cal_step == 2) y = UI_HEIGHT - 1 - CAL_MARGIN;
  lv_obj_set_pos(s_cal_dot, x - 9, y - 9);
  lv_label_set_text(s_cal_label, kText[s_cal_step]);
}

static void cal_event_cb(lv_event_t* e) {
  (void)e;
  uint16_t rx, ry;
  board_get_last_raw(&rx, &ry);
  s_cal_raw[s_cal_step][0] = rx;
  s_cal_raw[s_cal_step][1] = ry;
  if (++s_cal_step < 3) {
    cal_show_step_locked();
    return;
  }
  if (touch_calib_compute(s_cal_raw[0], s_cal_raw[1], s_cal_raw[2])) {
    touch_calib_save();
    xSemaphoreGive(s_cal_done);
  } else {
    s_cal_step = 0;
    cal_show_step_locked();
    lv_label_set_text(s_cal_label, "Taps were inconsistent. Tap the TOP-LEFT cross again");
  }
}

void ui_run_calibration(void) {
  touch_calib_invalidate();
  lvgl_port_lock(0);
  s_cal_step = 0;
  cal_show_step_locked();
  lv_obj_add_flag(s_splash, LV_OBJ_FLAG_HIDDEN);
  lv_obj_add_flag(s_main, LV_OBJ_FLAG_HIDDEN);
  lv_obj_remove_flag(s_cal, LV_OBJ_FLAG_HIDDEN);
  lv_obj_move_foreground(s_cal);
  lvgl_port_unlock();

  xSemaphoreTake(s_cal_done, portMAX_DELAY);

  lvgl_port_lock(0);
  lv_obj_add_flag(s_cal, LV_OBJ_FLAG_HIDDEN);
  lvgl_port_unlock();
  ESP_LOGI(TAG, "touch calibration done");
}

bool ui_recalibrate_requested(void) {
  const bool r = s_recal;
  s_recal = false;
  return r;
}

// ------------------------------------------------------------------ construction
void ui_init(lv_display_t* disp, ui_ask_cb_t on_ask) {
  (void)disp;
  s_on_ask = on_ask;
  s_cal_done = xSemaphoreCreateBinary();

  lvgl_port_lock(0);
  lv_obj_t* scr = lv_screen_active();
  lv_obj_set_style_bg_color(scr, lv_color_hex(COL_BG), 0);
  lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, 0);
  lv_obj_set_style_text_font(scr, &lv_font_montserrat_14, 0);
  lv_obj_remove_flag(scr, LV_OBJ_FLAG_SCROLLABLE);

  // --- splash / message ---
  s_splash = make_plain_container(scr, 0, 0, UI_WIDTH, UI_HEIGHT);
  s_splash_title = make_label(s_splash, "");
  lv_obj_align(s_splash_title, LV_ALIGN_TOP_MID, 0, 60);
  s_splash_text = make_label(s_splash, "");
  lv_obj_set_width(s_splash_text, UI_WIDTH - 40);
  lv_obj_set_style_text_align(s_splash_text, LV_TEXT_ALIGN_CENTER, 0);
  lv_obj_set_style_text_color(s_splash_text, lv_color_hex(COL_DIM), 0);
  lv_obj_align(s_splash_text, LV_ALIGN_TOP_MID, 0, 100);

  // --- main screen ---
  s_main = make_plain_container(scr, 0, 0, UI_WIDTH, UI_HEIGHT);
  lv_obj_add_flag(s_main, LV_OBJ_FLAG_HIDDEN);

  s_title = make_label(s_main, "WikiGPT");
  lv_obj_set_style_text_color(s_title, lv_color_hex(COL_ACCENT), 0);
  lv_obj_align(s_title, LV_ALIGN_TOP_LEFT, 6, 6);
  lv_obj_add_flag(s_title, LV_OBJ_FLAG_CLICKABLE);                       // long press = recalibrate touch
  lv_obj_add_event_cb(s_title, title_event_cb, LV_EVENT_LONG_PRESSED, NULL);

  s_status = make_label(s_main, "");
  lv_obj_set_width(s_status, 230);
  lv_obj_set_style_text_align(s_status, LV_TEXT_ALIGN_RIGHT, 0);
  lv_obj_set_style_text_color(s_status, lv_color_hex(COL_DIM), 0);
  lv_obj_align(s_status, LV_ALIGN_TOP_RIGHT, -6, 6);

  s_ta = lv_textarea_create(s_main);
  lv_obj_set_pos(s_ta, 4, 30);
  lv_obj_set_size(s_ta, 254, 36);
  lv_textarea_set_one_line(s_ta, true);
  lv_textarea_set_max_length(s_ta, 120);
  lv_textarea_set_placeholder_text(s_ta, "who is Barack Obama?");
  lv_obj_add_event_cb(s_ta, ta_event_cb, LV_EVENT_ALL, NULL);

  s_btn = lv_button_create(s_main);
  lv_obj_set_pos(s_btn, 262, 30);
  lv_obj_set_size(s_btn, 54, 36);
  lv_obj_t* bl = lv_label_create(s_btn);
  lv_label_set_text(bl, "Ask");
  lv_obj_center(bl);
  lv_obj_add_event_cb(s_btn, btn_event_cb, LV_EVENT_CLICKED, NULL);

  s_answer_box = make_plain_container(s_main, 4, 72, UI_WIDTH - 8, UI_HEIGHT - 76);
  lv_obj_add_flag(s_answer_box, LV_OBJ_FLAG_SCROLLABLE);
  lv_obj_set_scroll_dir(s_answer_box, LV_DIR_VER);
  s_answer = make_label(s_answer_box, "");
  lv_obj_set_width(s_answer, UI_WIDTH - 20);

  s_kb = lv_keyboard_create(s_main);
  lv_obj_set_size(s_kb, UI_WIDTH, 150);
  lv_obj_align(s_kb, LV_ALIGN_BOTTOM_MID, 0, 0);
  lv_keyboard_set_textarea(s_kb, s_ta);
  lv_obj_add_event_cb(s_kb, kb_event_cb, LV_EVENT_ALL, NULL);
  lv_obj_add_flag(s_kb, LV_OBJ_FLAG_HIDDEN);

  // --- calibration overlay (drawn above everything) ---
  s_cal = make_plain_container(scr, 0, 0, UI_WIDTH, UI_HEIGHT);
  lv_obj_set_style_bg_color(s_cal, lv_color_hex(COL_BG), 0);
  lv_obj_set_style_bg_opa(s_cal, LV_OPA_COVER, 0);
  lv_obj_add_flag(s_cal, LV_OBJ_FLAG_CLICKABLE);
  lv_obj_add_event_cb(s_cal, cal_event_cb, LV_EVENT_RELEASED, NULL);
  s_cal_label = make_label(s_cal, "");
  lv_obj_set_width(s_cal_label, UI_WIDTH - 80);
  lv_obj_set_style_text_align(s_cal_label, LV_TEXT_ALIGN_CENTER, 0);
  lv_obj_align(s_cal_label, LV_ALIGN_CENTER, 0, 0);
  s_cal_dot = lv_obj_create(s_cal);
  lv_obj_set_size(s_cal_dot, 18, 18);
  lv_obj_set_style_radius(s_cal_dot, LV_RADIUS_CIRCLE, 0);
  lv_obj_set_style_bg_color(s_cal_dot, lv_color_hex(0xFF3030), 0);
  lv_obj_set_style_border_width(s_cal_dot, 2, 0);
  lv_obj_set_style_border_color(s_cal_dot, lv_color_hex(0xFFFFFF), 0);
  lv_obj_remove_flag(s_cal_dot, LV_OBJ_FLAG_CLICKABLE);
  lv_obj_add_flag(s_cal, LV_OBJ_FLAG_HIDDEN);
  lvgl_port_unlock();
}

// ------------------------------------------------------------------ thread-safe setters
void ui_show_message(const char* title, const char* text) {
  lvgl_port_lock(0);
  lv_label_set_text(s_splash_title, title);
  lv_obj_align(s_splash_title, LV_ALIGN_TOP_MID, 0, 60);
  lv_label_set_text(s_splash_text, text);
  lv_obj_remove_flag(s_splash, LV_OBJ_FLAG_HIDDEN);
  lv_obj_add_flag(s_main, LV_OBJ_FLAG_HIDDEN);
  lv_obj_add_flag(s_cal, LV_OBJ_FLAG_HIDDEN);
  lvgl_port_unlock();
}

void ui_show_main(void) {
  lvgl_port_lock(0);
  lv_obj_add_flag(s_splash, LV_OBJ_FLAG_HIDDEN);
  lv_obj_remove_flag(s_main, LV_OBJ_FLAG_HIDDEN);
  lvgl_port_unlock();
}

void ui_set_status(const char* text) {
  lvgl_port_lock(0);
  lv_label_set_text(s_status, text);
  lvgl_port_unlock();
}

void ui_set_answer(const char* text) {
  lvgl_port_lock(0);
  lv_label_set_text(s_answer, text);
  lvgl_port_unlock();
}

void ui_set_busy(bool busy) {
  lvgl_port_lock(0);
  set_busy_locked(busy);
  lvgl_port_unlock();
}
