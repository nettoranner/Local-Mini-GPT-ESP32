// Touch calibration for the resistive XPT2046.
//
// The raw touch values are mapped to screen pixels with a linear map per axis. At first boot the user taps
// three crosses (top-left, top-right, bottom-left); from them we find out whether the axes are swapped and
// the scale/offset of each axis, so no manual flip/swap flags are needed. The result is stored in NVS.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Screen positions of the three calibration crosses.
#define CAL_MARGIN 24

bool touch_calib_load(void);        // from NVS (nvs_flash_init must have been called)
bool touch_calib_save(void);
bool touch_calib_valid(void);
void touch_calib_invalidate(void);  // forget the calibration (a rough default map is used until redone)

// Computes the calibration from the raw values read at the three crosses:
//   P1 = (CAL_MARGIN, CAL_MARGIN), P2 = (UI_WIDTH-1-CAL_MARGIN, CAL_MARGIN), P3 = (CAL_MARGIN, UI_HEIGHT-1-CAL_MARGIN)
// Returns false if the taps are inconsistent (too close together, wrong directions).
bool touch_calib_compute(const uint16_t raw1[2], const uint16_t raw2[2], const uint16_t raw3[2]);

// Raw touch value -> screen pixel (clamped to the screen).
void touch_calib_apply(uint16_t raw_x, uint16_t raw_y, int* screen_x, int* screen_y);

#ifdef __cplusplus
}
#endif
