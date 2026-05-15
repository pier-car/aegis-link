# =============================================================================
#  AEGIS-LINK :: tracking_system/CMakeLists.txt
# -----------------------------------------------------------------------------
#  Build the EKF tracking core. Optimised for x86_64 / aarch64 (WSL2 Ubuntu).
#
#  Usage:
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#     cmake --build build -j
# =============================================================================
cmake_minimum_required(VERSION 3.20)
project(aegis_tracker
        VERSION   1.0.0
        LANGUAGES CXX
        DESCRIPTION "AEGIS-LINK Extended Kalman Filter tracking core")

# ---------- Standard / build-type defaults ----------------------------------
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- Dependencies ----------------------------------------------------
find_package(Threads REQUIRED)
find_package(Eigen3  3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    # Fallback: use system Eigen headers under /usr/include/eigen3
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
              PATHS /usr/include/eigen3 /usr/local/include/eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found. Run install_all.sh.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN3_INCLUDE_DIR}")
endif()

# libzmq via pkg-config (most portable on Ubuntu).
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

# cppzmq — header-only C++ binding. Try CMake config first, fall back to header.
find_package(cppzmq QUIET)
if(NOT cppzmq_FOUND)
    find_path(CPPZMQ_INCLUDE_DIR zmq.hpp
              PATHS /usr/include /usr/local/include)
    if(NOT CPPZMQ_INCLUDE_DIR)
        message(FATAL_ERROR "cppzmq (zmq.hpp) not found. Run install_all.sh.")
    endif()
    add_library(cppzmq INTERFACE IMPORTED)
    target_include_directories(cppzmq INTERFACE "${CPPZMQ_INCLUDE_DIR}")
endif()

# ---------- Target ----------------------------------------------------------
add_executable(aegis_tracker main.cpp)

target_include_directories(aegis_tracker PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."          # for shared/messages.h
)

target_link_libraries(aegis_tracker PRIVATE
    Eigen3::Eigen
    PkgConfig::ZMQ
    cppzmq
    Threads::Threads
)

# ---------- Compiler flags --------------------------------------------------
# Release: aggressive but safe (no -ffast-math by default — Kalman covariances
# need IEEE-754 monotonicity).
target_compile_options(aegis_tracker PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -DNDEBUG -fno-plt>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -march=native>
        $<$<CONFIG:Debug>:-O0 -g3 -fsanitize=address,undefined>
    >
)
target_link_options(aegis_tracker PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Debug>>:-fsanitize=address,undefined>
)

# Eigen vectorisation hint
target_compile_definitions(aegis_tracker PRIVATE
    EIGEN_NO_DEBUG
    $<$<CONFIG:Release>:EIGEN_DONT_PARALLELIZE>
)

# ---------- Install ---------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS aegis_tracker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---------- Status banner ---------------------------------------------------
message(STATUS "----------------------------------------------------------")
message(STATUS "  AEGIS-LINK tracker  v${PROJECT_VERSION}")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen include   : ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  libzmq version  : ${ZMQ_VERSION}")
message(STATUS "----------------------------------------------------------")
