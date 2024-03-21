find_package(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw")
add_executable(checksum checksum.cpp)
set_property(TARGET checksum PROPERTY CXX_STANDARD 20)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX numa)
set_property(TARGET benchmark PROPERTY CXX_STANDARD 20)

add_executable(memory_bandwidth memory_bandwidth.cpp)
target_link_libraries(memory_bandwidth PRIVATE OpenMP::OpenMP_CXX dpu)
set_property(TARGET memory_bandwidth PROPERTY CXX_STANDARD 20)

if (SHIPPED_LIBDPU)
    target_link_libraries(checksum PRIVATE PkgConfig::DPU)
    target_link_libraries(benchmark PRIVATE PkgConfig::DPU)

else()
    target_compile_definitions(benchmark PUBLIC USE_DPU_NUMA=1)

    target_link_libraries(checksum PRIVATE  dpu dpuhw dpuverbose)
    target_link_libraries(benchmark PRIVATE dpu dpuhw dpuverbose)
endif()


