find_package(OpenMP)

add_executable(checksum checksum.cpp)
target_link_libraries(checksum PUBLIC dpu dpuhw dpuverbose)
set_property(TARGET checksum PROPERTY CXX_STANDARD 20)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PUBLIC dpu dpuhw dpuverbose numa OpenMP::OpenMP_CXX)
set_property(TARGET benchmark PROPERTY CXX_STANDARD 20)

