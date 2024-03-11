/*
 * Copyright (c) 2014-2017 - uPmem
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * An example of checksum computation with multiple tasklets.
 *
 * Every tasklet processes specific areas of the MRAM, following the "rake"
 * strategy:
 *  - Tasklet number T is first processing block number TxN, where N is a
 *    constant block size
 *  - It then handles block number (TxN) + (NxM) where M is the number of
 *    scheduled tasklets
 *  - And so on...
 *
 * The host is in charge of computing the final checksum by adding all the
 * individual results.
 */
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#include "checksum_common.h"

#define ELEMS_IN_CACHE (256 / 4)

__dma_aligned uint32_t DPU_CACHES[NR_TASKLETS][ELEMS_IN_CACHE];
__host dpu_results_t DPU_RESULTS;

__mram_noinit uint32_t DPU_BUFFER[BUFFER_SIZE];

/**
 * @fn main
 * @brief main function executed by each tasklet
 * @return the checksum result
 */
int main() {
    uint32_t tasklet_id = me();
    uint32_t * cache = DPU_CACHES[tasklet_id];
    dpu_result_t *result = &DPU_RESULTS.tasklet_result[tasklet_id];

    if (tasklet_id == 0) {
        DPU_RESULTS.nr_actual_tasklets = NR_TASKLETS;
    }

    uint32_t partial_checksum = checksum_init();

    const uint32_t n = DPU_BUFFER[0];

    for (uint32_t buffer_idx = tasklet_id * ELEMS_IN_CACHE; buffer_idx < n; buffer_idx += (NR_TASKLETS * ELEMS_IN_CACHE)) {

        /* load cache with current mram block. */
        mram_read(&DPU_BUFFER[buffer_idx], cache, ELEMS_IN_CACHE * 4);

        /* computes the checksum of a cached block */
        uint32_t end = ELEMS_IN_CACHE;
        if (n < buffer_idx + ELEMS_IN_CACHE) {
            end = n - buffer_idx;
        }

        for (uint32_t cache_idx = 0; cache_idx < end; cache_idx++) {
            partial_checksum = checksum_update(partial_checksum, buffer_idx + cache_idx, cache[cache_idx]);
        }
    }

    /* keep the 32-bit LSB on the 64-bit cycle counter */
    result->checksum = partial_checksum;

    printf("[%02d] n = 0x%08x Checksum = 0x%08x\n", tasklet_id, n, result->checksum);
    return 0;
}
