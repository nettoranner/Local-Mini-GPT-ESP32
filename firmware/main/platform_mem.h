// Memory allocation: on the ESP32-S3 big buffers go to PSRAM, hot ones to internal SRAM.
#pragma once
#include <cstddef>
#include <cstdlib>

#if defined(ESP_PLATFORM)
#include "esp_heap_caps.h"
inline void* AllocLarge(size_t n) {  // PSRAM
  return heap_caps_aligned_alloc(16, n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}
inline void* AllocFast(size_t n) {  // internal SRAM, any memory if that is not enough
  void* p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  return p ? p : heap_caps_aligned_alloc(16, n, MALLOC_CAP_8BIT);
}
inline void FreeMem(void* p) { heap_caps_free(p); }
#else
inline void* AllocLarge(size_t n) { return std::aligned_alloc(16, (n + 15) & ~size_t(15)); }
inline void* AllocFast(size_t n) { return std::aligned_alloc(16, (n + 15) & ~size_t(15)); }
inline void FreeMem(void* p) { std::free(p); }
#endif
