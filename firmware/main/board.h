// Board bring-up: SPI bus, ST7789 panel, XPT2046 touch and the LVGL port.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "lvgl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialises everything and returns the LVGL display (NULL on error). Also registers the touch as an
// LVGL pointer device that uses the calibration from touch_calib.h.
lv_display_t* board_init(void);

// Reads the touch controller. Returns true while pressed and stores the raw 12-bit values.
bool board_touch_read_raw(uint16_t* raw_x, uint16_t* raw_y);

// Last raw position seen while pressed (used by the calibration screen).
void board_get_last_raw(uint16_t* raw_x, uint16_t* raw_y);

#ifdef __cplusplus
}
#endif
