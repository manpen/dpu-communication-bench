find_package(OpenMP)

add_executable(checksum checksum.cpp)
target_link_libraries(checksum PUBLIC dpu dpuhw dpuverbose)
set_property(TARGET checksum PROPERTY CXX_STANDARD 20)


#add_executable(benchmark benchmark.cpp)
#target_link_libraries(benchmark PRIVATE dpu dpuhw dpuverbose PUBLIC OpenMP::OpenMP_CXX)
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${C_AVX512F_FLAGS} ${C_AVX512BW_FLAGS} ${C_CLFLUSHOPT_FLAGS}")

