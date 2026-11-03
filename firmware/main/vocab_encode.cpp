#include <cstring>

#include "vocab.h"

int VocabFind(const char* token) {
  for (int i = 0; i < kVocabSize; ++i)
    if (strcmp(kVocab[i], token) == 0) return i;
  return -1;
}

static int Utf8Len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

int VocabEncode(const char* text, uint16_t* out, int max_out, int unknown_index, int* n_unknown) {
  int n = 0, unk = 0;
  const char* s = text;
  while (*s) {
    int best = -1;
    size_t best_len = 0;
    for (int i = 0; i < kVocabSize; ++i) {
      const size_t len = strlen(kVocab[i]);
      if (len > best_len && strncmp(s, kVocab[i], len) == 0) { best = i; best_len = len; }
    }
    int idx = best;
    if (best < 0) {
      ++unk;
      best_len = Utf8Len(static_cast<unsigned char>(*s));
      idx = unknown_index;
    }
    s += best_len;
    if (idx < 0) continue;
    if (n == max_out) {  // window is full: drop the oldest symbol
      memmove(out, out + 1, sizeof(uint16_t) * (max_out - 1));
      --n;
    }
    out[n++] = static_cast<uint16_t>(idx);
  }
  if (n_unknown) *n_unknown = unk;
  return n;
}
