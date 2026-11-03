#include "board.h"

#include <string.h>

#include "board_config.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_touch_xpt2046.h"
#include "esp_log.h"
#include "esp_lvgl_port.h"
#include "touch_calib.h"

static const char* TAG = "board";

static esp_lcd_touch_handle_t s_touch;
static uint16_t s_last_raw_x, s_last_raw_y;

bool board_touch_read_raw(uint16_t* raw_x, uint16_t* raw_y) {
  uint16_t px[1], py[1];
  uint8_t cnt = 0;
  esp_lcd_touch_read_data(s_touch);
  const bool pressed = esp_lcd_touch_get_coordinates(s_touch, px, py, NULL, &cnt, 1);
  if (!pressed || cnt == 0) return false;
  *raw_x = px[0];
  *raw_y = py[0];
  return true;
}

void board_get_last_raw(uint16_t* raw_x, uint16_t* raw_y) {
  *raw_x = s_last_raw_x;
  *raw_y = s_last_raw_y;
}

// LVGL pointer callback: raw touch -> calibrated screen position.
static void touch_read_cb(lv_indev_t* indev, lv_indev_data_t* data) {
  (void)indev;
  static int last_x, last_y;
  uint16_t rx, ry;
  if (board_touch_read_raw(&rx, &ry)) {
    s_last_raw_x = rx;
    s_last_raw_y = ry;
    touch_calib_apply(rx, ry, &last_x, &last_y);
    data->state = LV_INDEV_STATE_PRESSED;
  } else {
    data->state = LV_INDEV_STATE_RELEASED;
  }
  data->point.x = last_x;
  data->point.y = last_y;
}

lv_display_t* board_init(void) {
  // ---- SPI bus ----
  const spi_bus_config_t bus_cfg = {
      .sclk_io_num = PIN_SPI_SCK,
      .mosi_io_num = PIN_SPI_MOSI,
      .miso_io_num = PIN_SPI_MISO,
      .quadwp_io_num = -1,
      .quadhd_io_num = -1,
      .max_transfer_sz = UI_WIDTH * 40 * sizeof(uint16_t),
  };
  if (spi_bus_initialize(LCD_SPI_HOST, &bus_cfg, SPI_DMA_CH_AUTO) != ESP_OK) {
    ESP_LOGE(TAG, "spi_bus_initialize failed");
    return NULL;
  }

  // ---- Display panel (ST7789) ----
  esp_lcd_panel_io_handle_t io = NULL;
  const esp_lcd_panel_io_spi_config_t io_cfg = {
      .dc_gpio_num = PIN_LCD_DC,
      .cs_gpio_num = PIN_LCD_CS,
      .pclk_hz = LCD_PIXEL_CLOCK_HZ,
      .lcd_cmd_bits = 8,
      .lcd_param_bits = 8,
      .spi_mode = 0,
      .trans_queue_depth = 10,
  };
  if (esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_SPI_HOST, &io_cfg, &io) != ESP_OK) {
    ESP_LOGE(TAG, "panel io init failed");
    return NULL;
  }
  esp_lcd_panel_handle_t panel = NULL;
  const esp_lcd_panel_dev_config_t panel_cfg = {
      .reset_gpio_num = PIN_LCD_RST,
      .rgb_ele_order = LCD_BGR_ORDER ? LCD_RGB_ELEMENT_ORDER_BGR : LCD_RGB_ELEMENT_ORDER_RGB,
      .bits_per_pixel = 16,
  };
  if (esp_lcd_new_panel_st7789(io, &panel_cfg, &panel) != ESP_OK) {
    ESP_LOGE(TAG, "ST7789 init failed");
    return NULL;
  }
  esp_lcd_panel_reset(panel);
  esp_lcd_panel_init(panel);
  esp_lcd_panel_invert_color(panel, LCD_INVERT_COLORS);

#if PIN_LCD_BL >= 0
  const gpio_config_t bl_cfg = {.pin_bit_mask = 1ULL << PIN_LCD_BL, .mode = GPIO_MODE_OUTPUT};
  gpio_config(&bl_cfg);
  gpio_set_level(PIN_LCD_BL, 1);
#endif

  // ---- Touch controller (XPT2046) on the same bus, its own chip select ----
  esp_lcd_panel_io_handle_t tp_io = NULL;
  const esp_lcd_panel_io_spi_config_t tp_io_cfg = ESP_LCD_TOUCH_IO_SPI_XPT2046_CONFIG(PIN_TOUCH_CS);
  if (esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_SPI_HOST, &tp_io_cfg, &tp_io) != ESP_OK) {
    ESP_LOGE(TAG, "touch io init failed");
    return NULL;
  }
  // x_max = y_max = 4095: the driver hands over (almost) raw 12-bit values; touch_calib.c maps them to pixels.
  const esp_lcd_touch_config_t tp_cfg = {
      .x_max = 4095,
      .y_max = 4095,
      .rst_gpio_num = GPIO_NUM_NC,
      .int_gpio_num = PIN_TOUCH_IRQ >= 0 ? PIN_TOUCH_IRQ : GPIO_NUM_NC,
      .levels = {.reset = 0, .interrupt = 0},
      .flags = {.swap_xy = 0, .mirror_x = 0, .mirror_y = 0},
  };
  if (esp_lcd_touch_new_spi_xpt2046(tp_io, &tp_cfg, &s_touch) != ESP_OK) {
    ESP_LOGE(TAG, "XPT2046 init failed");
    return NULL;
  }

  // ---- LVGL ----
  lvgl_port_cfg_t lvgl_cfg = ESP_LVGL_PORT_INIT_CONFIG();
  lvgl_cfg.task_stack = 8192;
  lvgl_cfg.task_affinity = 0;   // LVGL on core 0, the model runs on core 1
  if (lvgl_port_init(&lvgl_cfg) != ESP_OK) {
    ESP_LOGE(TAG, "lvgl_port_init failed");
    return NULL;
  }
  const lvgl_port_display_cfg_t disp_cfg = {
      .io_handle = io,
      .panel_handle = panel,
      .buffer_size = UI_WIDTH * 40,
      .double_buffer = true,
      .hres = UI_WIDTH,
      .vres = UI_HEIGHT,
      .monochrome = false,
      .color_format = LV_COLOR_FORMAT_RGB565,
      .rotation = {.swap_xy = LCD_SWAP_XY, .mirror_x = LCD_MIRROR_X, .mirror_y = LCD_MIRROR_Y},
      .flags = {.buff_dma = true, .swap_bytes = true},
  };
  lv_display_t* disp = lvgl_port_add_disp(&disp_cfg);
  if (!disp) {
    ESP_LOGE(TAG, "lvgl_port_add_disp failed");
    return NULL;
  }
  esp_lcd_panel_disp_on_off(panel, true);

  // Touch as an LVGL pointer device with our own calibration.
  lvgl_port_lock(0);
  lv_indev_t* indev = lv_indev_create();
  lv_indev_set_type(indev, LV_INDEV_TYPE_POINTER);
  lv_indev_set_read_cb(indev, touch_read_cb);
  lv_indev_set_display(indev, disp);
  lvgl_port_unlock();

  ESP_LOGI(TAG, "display %dx%d and touch ready", UI_WIDTH, UI_HEIGHT);
  return disp;
}
