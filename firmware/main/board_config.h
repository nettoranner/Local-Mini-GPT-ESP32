// Board wiring and display settings (ESP32-S3 N16R8 + ST7789V 2.8" 240x320 + XPT2046 touch).
// This is the wiring that was verified with the Arduino test sketch. Both devices share one SPI bus.
// Avoid GPIO 35/36/37 (octal PSRAM), 26-32 (flash), 19/20 (USB) on the N16R8.
#pragma once

// ---- SPI bus (shared by the display and the touch controller) ----
#define PIN_SPI_SCK   12
#define PIN_SPI_MOSI  11
#define PIN_SPI_MISO  13   // the touch controller answers on this line
#define LCD_SPI_HOST  SPI2_HOST

// ---- Display ----
#define PIN_LCD_CS    10
#define PIN_LCD_DC     9
#define PIN_LCD_RST    8
#define PIN_LCD_BL    -1   // -1: backlight (LED pin) is wired to 3V3; or a GPIO number to switch it on
#define LCD_PIXEL_CLOCK_HZ (20 * 1000 * 1000)   // lower it (e.g. 10 MHz) if the picture is noisy

// ---- Touch (XPT2046) ----
#define PIN_TOUCH_CS  14
#define PIN_TOUCH_IRQ 15   // -1 if T_IRQ is not connected

// ---- Orientation ----
// The panel is 240x320 (portrait). The UI is landscape 320x240, so the panel is rotated with MADCTL.
// If the picture is mirrored or upside down, change LCD_MIRROR_X / LCD_MIRROR_Y.
#define UI_WIDTH      320
#define UI_HEIGHT     240
#define LCD_SWAP_XY     1
#define LCD_MIRROR_X    1
#define LCD_MIRROR_Y    0

// ---- Colours ----
// Your Arduino test needed no colour inversion. If the picture looks like a negative, set this to 1.
#define LCD_INVERT_COLORS 0
// If red and blue are swapped, set this to 1.
#define LCD_BGR_ORDER     0
