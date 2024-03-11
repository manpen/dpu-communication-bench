// This code is derived from the original code in the upmem-sdk repository.
// But we are using a different checksum and also randomize the values.

extern "C" {
#include <dpu.h>
#include "../common/checksum_common.h"
}

#include <iostream>
#include <random>
#include <vector>

#define DPU_BINARY "checksum_dpu"

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RESET "\x1b[0m"

constexpr size_t DPUS_PER_RANK = 64;

using T = uint32_t;
using data_buffer_t = std::vector<T>;
using data_buffers_t = std::vector<data_buffer_t>;

void transfer_input_to_dpus(dpu_set_t dpu_set, bool broadcast, const data_buffers_t &buffers);

data_buffer_t generate_buffer(std::mt19937 &urng, size_t n) {
    data_buffer_t buffer(n);

    for (auto &x: buffer)
        x = urng();
    buffer[0] = n;

    return buffer;
}

template <typename It>
T compute_checksum(It begin, It end) {
    T checksum = checksum_init();
    for (auto it = begin; it != end; ++it) {
        checksum = checksum_update(checksum, it - begin, *it);
    }
    return checksum;
}


void transfer_input_to_dpus(dpu_set_t dpu_set, bool broadcast, const data_buffers_t &buffers) {
    const size_t nr_bytes = buffers[0].size() * sizeof(uint32_t);

    if (broadcast) {
        DPU_ASSERT(dpu_broadcast_to(dpu_set, XSTR(DPU_BUFFER), 0, reinterpret_cast<const void *>(buffers[0].data()), nr_bytes,
                                    DPU_XFER_DEFAULT));
    } else {
        dpu_set_t rank;

        DPU_RANK_FOREACH(dpu_set, rank) {

            dpu_set_t dpu;
            uint32_t dpu_id;
            DPU_FOREACH(rank, dpu, dpu_id) {
                // const cast okay, since libdpu only accesses to read
                DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(buffers[dpu_id].data())));
            }

            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, XSTR(DPU_BUFFER), 0,
                                     nr_bytes, DPU_XFER_DEFAULT));
        }
    }
}

std::vector<dpu_results_t> fetch_results_from_dpu(dpu_set_t dpu_set) {
    uint32_t nr_dpus;
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));

    std::vector<dpu_results_t> results(nr_dpus);

    {
        dpu_set_t dpu;
        uint32_t each_dpu;
        DPU_FOREACH(dpu_set, dpu, each_dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &results[each_dpu]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, XSTR(DPU_RESULTS), 0,
                                 sizeof(dpu_results_t), DPU_XFER_DEFAULT));
    }

    return results;
}

uint32_t combine_partial_dpu_results(const dpu_results_t& dpu_result) {
    uint32_t checksum = checksum_init();
    for(uint32_t i = 0; i < dpu_result.nr_actual_tasklets; ++i) {
        checksum = checksum_combine(checksum, dpu_result.tasklet_result[i].checksum);
    }
    return checksum;
}

bool run_test(dpu_set_t dpu_set, bool broadcast,
              data_buffers_t &buffers) {

    std::cout << "Run tests with n=" << buffers[0].size() << " and broadcast=" << broadcast;

    transfer_input_to_dpus(dpu_set, broadcast, buffers);

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    if constexpr (0){
        dpu_set_t dpu;
        DPU_FOREACH(dpu_set, dpu) { DPU_ASSERT(dpu_log_read(dpu, stdout)); }
    }

    auto dpus_results = fetch_results_from_dpu(dpu_set);

    size_t nr_mismatches = 0;
    for(size_t idx = 0; idx < dpus_results.size(); ++idx) {
        const auto dpu_idx = idx % DPUS_PER_RANK;
        const auto rank_idx = idx / DPUS_PER_RANK;
        const auto& buffer = buffers[broadcast ? 0 : dpu_idx];

        const auto expected_checksum = compute_checksum(buffer.cbegin(), buffer.cend());

        const auto dpu_checksum = combine_partial_dpu_results(dpus_results[idx]);

        if (expected_checksum == dpu_checksum) {
            continue;
        }

        if (nr_mismatches == 0) {
            std::cout <<  ANSI_COLOR_RED " [MISMATCHES: " << nr_mismatches << "]" ANSI_COLOR_RESET  "\n" ;
        }

        std::cerr << ANSI_COLOR_RED "Mismatch for idx=" << idx << " (DPU: " << dpu_idx << ", Rank: " << rank_idx << "). Expected=" << expected_checksum << " Got=" << dpu_checksum << ANSI_COLOR_RESET "\n";
        nr_mismatches++;
    }

    if (nr_mismatches == 0) {
        std::cout << ANSI_COLOR_GREEN " [OK]" ANSI_COLOR_RESET  "\n" ;
    }

    return nr_mismatches == 0;
}



int main() {
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    uint32_t theoretical_checksum, dpu_checksum;
    uint32_t dpu_cycles;
    bool status = true;

    //DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &dpu_set));
    DPU_ASSERT(dpu_alloc_ranks(1, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    constexpr size_t max_n = (1 << 20) / sizeof(T);

    for(size_t n = 8; n < max_n; n *= 3) {
        data_buffers_t buffers;
        {
            buffers.reserve(DPUS_PER_RANK);
            std::mt19937 rng(std::random_device{}());
            for (size_t i = 0; i < DPUS_PER_RANK; ++i) {
                buffers.push_back(generate_buffer(rng, n));
            }
        }

        if (!run_test(dpu_set, false, buffers)) return 1;
        if (!run_test(dpu_set, true, buffers)) return 1;
    }

    return 0;
}
