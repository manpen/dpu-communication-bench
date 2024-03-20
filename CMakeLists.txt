cmake_minimum_required(VERSION 3.13)
project(upmem-communications)

option(SHIPPED_LIBDPU "Build against the precompiled SDK")


if (SHIPPED_LIBDPU)
    set(PKG_CONFIG_EXECUTABLE "dpu-pkg-config")
    find_package(PkgConfig REQUIRED)
    pkg_search_module(DPU REQUIRED IMPORTED_TARGET dpu)
else()
    add_subdirectory(upmem-libdpu)
endif()

add_subdirectory(host)
