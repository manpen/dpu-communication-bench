#ifndef __COMMON_H__
#define __COMMON_H__

#define XSTR(x) STR(x)
#define STR(x) #x

/* DPU variable that will be read of write by the host */
#define DPU_BUFFER dpu_mram_buffer
#define DPU_CACHES dpu_wram_caches
#define DPU_RESULTS dpu_wram_results

/* Size of the buffer on which the checksum will be performed */
#define BUFFER_SIZE (60 << 20) / 4

/* Structure used by both the host and the dpu to communicate information */

#include <stdint.h>

typedef struct {
  uint32_t checksum;
} dpu_result_t;

typedef struct {
  uint32_t nr_actual_tasklets;
  dpu_result_t tasklet_result[24];
} dpu_results_t;

static uint32_t hash(uint32_t x) {
  // that's the hash function used in XORSHIFT32
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

static inline uint32_t murmur_32_scramble(uint32_t k) {
  k *= 0xcc9e2d51;
  k = (k << 15) | (k >> 17);
  k *= 0x1b873593;
  return k;
}

// The following checksum accounts for the values of the elements
// as well as their position, with the twist that we are free to add the
// elements in an arbitrary order, as long as we add each element only
// once and with its original index.
//
// # Example
// uint32_t checksum = checksum_init();
// for (uint32_t i = 0; i < size; i++) {
//  checksum = checksum_update(checksum, i, buffer[i]);
// }
//
// // produces the same results as
//
// uint32_t even = checksum_init();
// uint32_t odd  = checksum_init();
// for (uint32_t i = 0; i < size; i++) {
//  if (i % 2 == 0) {
//    even = checksum_update(even, i, buffer[i]);
//  } else {
//    odd = checksum_update(odd, i, buffer[i]);
//  }
// }
// checksum = checksum_combine(even, odd);
static uint32_t checksum_init() { return 0; }
static uint32_t checksum_update(uint32_t state, uint32_t index,
                                uint32_t value) {
  uint32_t tmp = value ^ index;

  index = hash(index);
  value = murmur_32_scramble(value);
  while (index > 0) {
    tmp ^= value >> (index % 3);
    index /= 3;

    tmp ^= value << (index % 7);
    index /= 7;
  }

  state += tmp;
  return state;
};
static uint32_t checksum_combine(uint32_t state1, uint32_t state2) {
  return state1 + state2;
}

#endif /* __COMMON_H__ */
