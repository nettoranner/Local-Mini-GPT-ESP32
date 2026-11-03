// LVGL user interface: splash/message screen, touch calibration, and the main screen
// (question field + on-screen keyboard + answer area). All functions are thread-safe (they take the LVGL lock).
#pragma once
#include <stdbool.h>

#include "lvgl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Called (in the LVGL task, must not block) when the user submits a question.
// Return true if the question was accepted.
typedef bool (*ui_ask_cb_t)(const char* question);

void ui_init(lv_display_t* disp, ui_ask_cb_t on_ask);

// Full-screen message (boot progress, errors).
void ui_show_message(const char* title, const char* text);
// Show the main screen.
void ui_show_main(void);

void ui_set_status(const char* text);   // one line in the title bar
void ui_set_answer(const char* text);   // replaces the answer text
void ui_set_busy(bool busy);            // disables the Ask button while the model is thinking

// Runs the 3-tap touch calibration, stores the result and returns when done. Call it from a normal task,
// NOT from an LVGL callback (it waits for the taps).
void ui_run_calibration(void);
// True once after the user long-presses the title (request to recalibrate).
bool ui_recalibrate_requested(void);

#ifdef __cplusplus
}
#endif
