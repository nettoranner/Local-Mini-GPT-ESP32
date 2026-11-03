/*
Used ST7789V-2.8 TFT Touch and ESP32-S3 N16R8
*/

// ESP32-S3 N16R8 + ST7789V 2.8" 240x320 + XPT2046 touch
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <XPT2046_Touchscreen.h>

// Shared SPI bus 
#define PIN_SCK 12
#define PIN_MOSI 11
#define PIN_MISO 13

// Display 
#define TFT_CS 10
#define TFT_DC 9
#define TFT_RST 8

// Touch
#define TOUCH_CS 14
#define TOUCH_IRQ 15      // set to 255 if T_IRQ is not connected

// Set to true if the background shows white instead of black
#define INVERT_COLORS false

// Touch orientation 
#define FLIP_X true      // mirror horizontally
#define FLIP_Y true      // mirror vertically
#define SWAP_XY false    // swap X and Y axes

// Touch calibration (adjust using raw values from Serial Monitor)
#define TS_MINX 200
#define TS_MAXX 3800
#define TS_MINY 200
#define TS_MAXY 3800

Adafruit_ST7789 tft(&SPI, TFT_CS, TFT_DC, TFT_RST);
XPT2046_Touchscreen ts(TOUCH_CS, TOUCH_IRQ);

// RGB565 colors
#define C_BLACK 0x0000
#define C_WHITE 0xFFFF
#define C_RED 0xF800
#define C_GREEN 0x07E0
#define C_BLUE 0x001F
#define C_YELLOW 0xFFE0
#define C_GRAY 0x7BEF

const int BTN_H = 40;   // CLEAR button height

// Draw the UI: black canvas with a CLEAR button on top
void drawUI() {
  tft.fillScreen(C_BLACK);
  tft.fillRect(0, 0, tft.width(), BTN_H, C_GRAY);
  tft.setTextSize(2);
  tft.setTextColor(C_WHITE);
  tft.setCursor(80, 12);
  tft.print("CLEAR");
}

// Convert raw touch coordinates to screen coordinates
bool getTouchXY(int &x, int &y) {
  if (!ts.touched()) return false;

  TS_Point p = ts.getPoint();
  Serial.printf("raw x=%d y=%d z=%d\n", p.x, p.y, p.z);

  int rx = p.x;
  int ry = p.y;
  if (SWAP_XY) { int t = rx; rx = ry; ry = t; }

  x = map(rx, TS_MINX, TS_MAXX, 0, tft.width());
  y = map(ry, TS_MINY, TS_MAXY, 0, tft.height());

  if (FLIP_X) x = tft.width()  - 1 - x;
  if (FLIP_Y) y = tft.height() - 1 - y;

  x = constrain(x, 0, tft.width()  - 1);
  y = constrain(y, 0, tft.height() - 1);
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== Start ===");

  // Pull both CS lines HIGH so devices don't interfere on the shared bus
  pinMode(TFT_CS, OUTPUT);   digitalWrite(TFT_CS, HIGH);
  pinMode(TOUCH_CS, OUTPUT); digitalWrite(TOUCH_CS, HIGH);

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI);

  tft.init(240, 320);
  tft.setSPISpeed(20000000);
  tft.invertDisplay(INVERT_COLORS);
  tft.setRotation(0);
  Serial.println("Display init OK");

  ts.begin();
  ts.setRotation(0);
  Serial.println("Touch init OK");

  // Color test
  tft.fillScreen(C_RED); delay(600);
  tft.fillScreen(C_GREEN); delay(600);
  tft.fillScreen(C_BLUE); delay(600);

  drawUI();
  Serial.println("Draw on the screen");
}

void loop() {
  int x, y;
  if (!getTouchXY(x, y)) return;

  if (y < BTN_H) {
    drawUI();      // CLEAR pressed
    delay(200);
  } else {
    tft.fillCircle(x, y, 3, C_YELLOW);
  }
}