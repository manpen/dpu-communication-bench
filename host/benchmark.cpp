#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <regex>

#include <cassert>

#include "timer.hpp"

extern "C" {
#include <dpu.h>
#ifdef USE_DPU_NUMA
#include <dpu_rank.h>
#endif
#include <numa.h>
}

const size_t nr_ranks = 32;
const size_t nr_dpus_per_rank = 64;
const char *binary = "./checksum_dpu";
using T = uint32_t;

std::vector<T *> allocate_buffers(size_t total_elements) {
  Timer timer("Allocation");
  const auto nr_numa_nodes = numa_num_configured_nodes();
  const auto elements_per_numa_node = total_elements / nr_numa_nodes;

  std::vector<T *> buffer(nr_numa_nodes);

  for (int i = 0; i < nr_numa_nodes; ++i) {
    const auto bytes = elements_per_numa_node * sizeof(T);

    buffer[i] = reinterpret_cast<T *>((((uintptr_t) numa_alloc_onnode(bytes + 128, i) + 63) / 64) * 64);

    if (buffer[i] == nullptr) {
      std::cerr << "Failed to allocate buffer on node " << i << "\n";
      abort();
    }

#pragma omp parallel for
    for (size_t j = 0; j < elements_per_numa_node; ++j) {
      buffer[i][j] = static_cast<T>(j);
    }

    std::cout << "Allocated " << (bytes >> 20) << " MiB on node " << i << "\n";
  }

  return buffer;
}

enum class Mode { Scatter, Scatter2Per8, Scatter4Per8, Broadcast, ControllerBroadcast, Gather };
const char* mode_to_string(Mode mode) {
    if (mode == Mode::Broadcast) {
        return "Broadcast";
    }

    if (mode == Mode::ControllerBroadcast) {
        return "ControllerBroadcast";
    }

    if (mode == Mode::Gather) {
        return "Gather";
    }

    if (mode == Mode::Scatter) {
        return "Scatter";
    }

    if (mode == Mode::Scatter2Per8) {
        return "Scatter2Per8";
    }

    if (mode == Mode::Scatter4Per8) {
        return "Scatter4Per8";
    }

    abort();
}


void benchmark(dpu_set_t dpu_set,
               bool aligned,
               std::vector<T *> buffers,
               size_t nr_elem_per_dpu, Mode mode,
               const char *profile) {

  const uint32_t nr_dpus = [&] {
    uint32_t tmp;
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &tmp));
    return tmp;
  }();

  const uint32_t nr_numa_nodes = static_cast<size_t>(buffers.size());

  int numa_rank_offset = rand();
  (void)numa_rank_offset;

  Timer timer("Transfer", nr_dpus * nr_elem_per_dpu * sizeof(T));

  struct dpu_set_t rank, dpu;
  uint32_t rank_id, dpu_id;

  DPU_RANK_FOREACH(dpu_set, rank, rank_id) {
#ifdef USE_DPU_NUMA
    const auto rank_numa_node = rank.list.ranks[0]->numa_node;
#else
    const auto rank_numa_node = (numa_rank_offset + rank_id) % nr_numa_nodes;
#endif

    DPU_FOREACH(rank, dpu, dpu_id) {
      T *first;
      switch (mode) {
      case Mode::Scatter:
      case Mode::Scatter2Per8:
      case Mode::Scatter4Per8:
      case Mode::Gather:
      {
        if (mode == Mode::Scatter2Per8 &&  (rank_id + dpu_id) % 2 != 0) continue;
        if (mode == Mode::Scatter4Per8 &&  (rank_id + dpu_id) % 4 != 0) continue;
        first = buffers[rank_numa_node];
        buffers[rank_numa_node] += nr_elem_per_dpu;
        break;
      }

      case Mode::Broadcast: {
        first = buffers[0];
        break;
      }

      case Mode::ControllerBroadcast: {
        first = buffers[rank_numa_node];
        break;
      }
      }

      DPU_ASSERT(dpu_prepare_xfer(dpu, first));
    }

    const auto bytes_per_dpu = nr_elem_per_dpu * sizeof(T);
    DPU_ASSERT(dpu_push_xfer(
        rank, mode == Mode::Gather ? DPU_XFER_FROM_DPU : DPU_XFER_TO_DPU,
        "dpu_mram_buffer", 0, bytes_per_dpu, DPU_XFER_ASYNC));
  }

  DPU_ASSERT(dpu_sync(dpu_set));

  const auto elapsed = timer.seconds_since_start();
  auto gbs =
      ((double)nr_dpus * nr_elem_per_dpu * sizeof(T)) / (1 << 30) / elapsed;

  std::cerr << "{" //
               "\"mode\": \""
            << mode_to_string(mode)
            << "\", " //
               "\"seconds\": "
            << elapsed
            << ", " //
               "\"dpus\": "
            << nr_dpus
            << ", " //
               "\"numa_nodes\": "
            << nr_numa_nodes
            << ", " //
               "\"bytes_per_dpu\": "
            << (nr_elem_per_dpu * sizeof(T))
            << ", " //
               "\"gbs\": "
            << gbs
            << ", " //
               "\"aligned\": "
            << aligned <<
               ", " //
               "\"profile\": \""
            << profile << "\"}\n";

  timer.hide();
}

dpu_set_t alloc_dpus(const char *profile) {
  struct dpu_set_t set;
  uint32_t nr_dpus;
  std::cout << "Profile: " << profile << "\n";

  DPU_ASSERT(dpu_alloc_ranks(nr_ranks, profile, &set));

  if (set.list.nr_ranks != nr_ranks) {
    std::cout << "Expected " << nr_ranks << " ranks, got " << set.list.nr_ranks
              << "\n";
    abort();
  }

  DPU_ASSERT(dpu_load(set, binary, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  std::cout << "Got " << nr_dpus << " DPUs\n";

  if (nr_dpus != nr_ranks * nr_dpus_per_rank) {
    std::cout << "Expected " << nr_ranks * nr_dpus_per_rank << " DPUs, got "
              << nr_dpus << "\n";
    abort();
  }

  return set;
}

std::vector<Mode> fetch_benchmark_modes(const char* regex) {
    const std::regex re(regex);
    std::vector<Mode> result;

    auto add_if_match = [&] (Mode mode) {
        std::smatch match;
        const auto name = std::string(mode_to_string(mode));
        if (std::regex_match(name, match, re)) {
            result.push_back(mode);
        }
    };

    add_if_match(Mode::Scatter);
    add_if_match(Mode::Scatter2Per8);
    add_if_match(Mode::Scatter4Per8);
    add_if_match(Mode::Broadcast);
    add_if_match(Mode::ControllerBroadcast);
    add_if_match(Mode::Gather);

    if (result.empty()) {
        std::cerr << "Pattern does not match any benchmarks\n";
        abort();
    }

    return result;
}


int main(int argc, char* argv[]) {
  if (numa_available() == -1) {
    std::cerr << "No NUMA support\n";
    abort();
  }

  const auto modes = fetch_benchmark_modes(argc > 1 ? argv[1] : ".*");

#ifdef USE_DPU_NUMA
  std::cout << "Using NUMA infos of each DPU\n";
#endif
  const size_t max_elems_per_dpu = (60 << 20) / sizeof(T);
  const auto total_elements = nr_ranks * nr_dpus_per_rank * max_elems_per_dpu;
  const auto buffers = allocate_buffers(total_elements);
  std::vector<T*> unaligned_buffers;
  for(auto* p : buffers) {
      unaligned_buffers.push_back(p + 1);
  }

  for (int i = 0; i < 3; ++i) {
      for (int nrThreadPerPool = 1; nrThreadPerPool <= 8;
           nrThreadPerPool *= 2) {

        std::string profile =
            "nrThreadPerPool=" + std::to_string(nrThreadPerPool);

        auto set = alloc_dpus(profile.c_str());

        for (size_t n = 16; true; n *= 2) {
          n = std::min(n, max_elems_per_dpu);
          for (auto mode : modes) {
              for(int aligned = 0; aligned <= 1; ++aligned) {
                  benchmark(set, aligned, aligned ? buffers : unaligned_buffers, n, mode, profile.c_str());
              }
          }
          if (n == max_elems_per_dpu) {
            break;
          }
        }

        DPU_ASSERT(dpu_free(set));
      }
  }

  std::cerr << "\n";
}
