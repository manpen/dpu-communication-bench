#include <vector>
#include "immintrin.h"
#include "timer.hpp"

using T = __m512i;
constexpr size_t nr_bytes = 32llu << 30;
constexpr size_t nr_elems = nr_bytes / sizeof(T);
constexpr size_t nr_align = sizeof(T);


void benchmark_interleaved_avx512(T* output, size_t steps) {
    BandwidthTimer timer(std::string("Interleaved [AVX512] steps=") + std::to_string(steps), nr_bytes);

#pragma omp parallel for schedule(static, steps)
    for(size_t i = 0; i < nr_elems; ++i) {
        const auto simd = _mm512_set1_epi64(i);
        _mm512_stream_si512(output + i, simd);
    }
}

void benchmark_interleaved_avx2(T* output, size_t steps) {
    auto* output256 = reinterpret_cast<__m256i*>(output);
    BandwidthTimer timer(std::string("Interleaved [ AVX2 ] steps=") + std::to_string(steps), nr_bytes);


#pragma omp parallel
    {
        const auto start = std::chrono::high_resolution_clock::now();


#pragma for schedule(static, steps)
        for(size_t i = 0; i < nr_elems * 2; ++i) {
            const auto simd = _mm256_set1_epi32(i);
            _mm256_stream_si256(output256 + i, simd);
        }

#pragma omp master
        {
            const auto stop  = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> diff = stop - start;
            std::cout << "AVX2 steps=" << steps << ": " << (nr_bytes >> 30) / diff.count() << "\n";
        };
    };


}


int main() {
    std::vector<uint8_t> data(nr_bytes + nr_align);
    T* aligned_buffer = reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(data.data()) + nr_align - 1) & ~(nr_align - 1));

    std::cout << "Allocated" << std::endl;

    for(int i=0 ; i < 10; ++i) {
        benchmark_interleaved_avx512(aligned_buffer, 1);
        benchmark_interleaved_avx512(aligned_buffer, 1 << 20);
        benchmark_interleaved_avx2(aligned_buffer, 1);
        benchmark_interleaved_avx2(aligned_buffer, 1 << 20);
    }

    return 0;
}