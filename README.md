# Mini-WikiGPT on ESP32-S3

A 12M parameter character model that answers questions **offline**, on an ESP32-S3 N16R8 with a
2.8" ST7789V touch screen. You type the question on the on-screen keyboard, the answer is written to the
screen character by character.

* Model: window 256 characters, vocabulary 98, one hidden layer of 480 units, int8 weights (~12 MB).
* The model lives in its own flash partition (`model`) and is **read in place** through the flash cache;
  only ~60 KB of hot data is copied to RAM.
* No TensorFlow Lite on the device. The firmware has one small inference routine (`main/cmlp_model.cpp`).

![Project Preview](assets/preview.png)

## How the model works (short)

The network is `one-hot(256 x 98) -> Dense(480) + ReLU -> Dense(98)`. A one-hot input times a matrix is
just *a sum of selected rows*, so layer 1 needs only 256 additions of 480-byte table rows instead of
12 million multiplications. The table is stored transposed (`[256*98][480]`) so each row is contiguous in
flash. Layer 2 runs in int32 and produces float logits; the next character is the arg-max (greedy).

It was trained on `"<question>\n<answer><END>"`, so the firmware fills the window with the question and
a newline, then generates until the END symbol.

## Hardware and wiring

| Signal | GPIO |
|---|---|
| SPI SCK / MOSI / MISO (shared) | 12 / 11 / 13 |
| LCD CS / DC / RST | 10 / 9 / 8 |
| Touch CS / IRQ | 14 / 15 |
| LCD backlight (LED) | 3V3 |

All pins and the screen orientation are in `main/board_config.h`. On the N16R8 avoid GPIO 35-37 (PSRAM),
26-32 (flash) and 19/20 (USB).

## Build and flash

Requires ESP-IDF 5.4 (the ST7789 driver is part of `esp_lcd` there) and internet access for the component
manager (LVGL 9.2, `esp_lvgl_port`, `esp_lcd_touch_xpt2046`).


#### 1. model.bin from the .tflite made by training/export_tflite.py
```bash
python -m pip install -r requirements.txt
python firmware/tools/export_model_bin.py char_mlp_int8.tflite --ctx 256 --vocab 98 --verify
```

#### 2. reference vectors for the boot self-test
```bash
python firmware/tools/make_golden.py model/model.bin
```

#### 3. build + flash (writes bootloader, app AND the model; ~12 MB, use a fast baud rate)
```bash
idf.py set-target esp32s3
idf.py build
idf.py -p <port-name> flash monitor
```

* `--verify` compares the exported hidden layer with the original `.tflite` on random windows: the
  difference must be **0**. If it is not, do not flash.
* `idf.py app-flash` flashes only the application (fast); the model partition stays untouched.
* Only the model, e.g. after retraining: `python -m esptool --chip esp32s3 -p COM5 write_flash 0x310000 model/model.bin`
* If you copy this project over an older one, delete the old `sdkconfig` and `build/` first.

## Vocabulary

`main/vocab.cpp` is generated and already matches the vocabulary of `training/build_qa.py`:
index 0 = padding (`\x00`), 1..95 = ASCII 32..126, 96 = newline, 97 = END (`\x03`). To make sure it is
identical to the `vocab.json` of your training run:

```
python tools/export_vocab.py --json vocab.json --check
```

## First boot

1. **Touch calibration.** Tap the three red crosses (top-left, top-right, bottom-left). The code works out
   whether the axes are swapped and the scale of each axis, and stores it in NVS. To redo it later,
   long-press the "WikiGPT" title.
2. The model is mapped, its CRC is checked (a progress percentage is shown) and the **self-test** compares
   the hidden layer and logits with the PC reference. The status line shows `Ready (self-test OK)`.
3. Tap the text field, type a question, press the keyboard's check mark (or **Ask**).

The model answers best in the wording it was trained on (`who is Barack Obama?`, `tell me about ...`).
It memorises answers: for a person it has not seen, it produces plausible but invented text.

## If something looks wrong

| Symptom | What to change (`main/board_config.h`) |
|---|---|
| Picture mirrored / upside down | `LCD_MIRROR_X`, `LCD_MIRROR_Y` |
| Colours look like a negative | `LCD_INVERT_COLORS` = 1 |
| Red and blue swapped | `LCD_BGR_ORDER` = 1 |
| Noise / flicker | lower `LCD_PIXEL_CLOCK_HZ` |
| Touch does nothing | try `PIN_TOUCH_IRQ` = -1 (polling only) |
| `no CMLP model in the partition` | flash the model: `idf.py flash` with `model/model.bin` present |
| `golden data belongs to another model` | run `tools/make_golden.py` again, rebuild |
| Build error about `esp_lcd_new_panel_st7789` | ESP-IDF >= 5.5: add the `espressif/esp_lcd_st7789` component |

## Layout

```
main/main.cpp        boot sequence
main/board.c         SPI bus, ST7789, XPT2046, LVGL port
main/touch_calib.c   touch calibration (stored in NVS)
main/ui.c            LVGL screens: message, calibration, question + keyboard + answer
main/engine.cpp      background task that answers questions
main/generate.cpp    question -> answer (pure logic)
main/cmlp_model.cpp  the int8 model: flash-resident table, forward pass, CRC
main/self_test.cpp   boot self-test against golden_data.h
main/vocab.cpp       generated symbol table
tools/               export_model_bin.py, make_golden.py, export_vocab.py, cmlp.py (Python twin of the C++ code)
```

## Data sources and license

The training answers are derived from the English Wikipedia
(dataset `wikimedia/wikipedia`, snapshot `20231101.en`, on Hugging Face).
Wikipedia text is available under the
[Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0). Authors of the articles are listed in each article's history on
https://en.wikipedia.org.

Changes made: only the first one or two sentences of each article are used,
text in parentheses is removed, and characters are converted to plain ASCII.

The list of article titles was built from the Wikimedia pageviews API (2023).

Code in this repository is released under the MIT License.
Model weights trained on Wikipedia-derived text are released under CC BY-SA 4.0.
