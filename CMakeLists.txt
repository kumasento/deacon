cmake_minimum_required(VERSION 3.10)

project(deacon LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(WITH_MAXCOMPILER "Build with MaxCompiler." ON)

# -------------------- Configuration --------------------
if (WITH_MAXCOMPILER)
  # MaxCompiler
  set(MAXCOMPILER_BUILD_DIR "/tmp/maxcompiler_builds" CACHE STRING "Where the MaxCompiler build outcome should be placed.")
  if(DEFINED ENV{MAXCOMPILER_PATH})
    message(STATUS "MAXCOMPILER_PATH: $ENV{MAXCOMPILER_PATH}")
  else()
    message(FATAL_ERROR "Environment variable MAXCOMPILER_PATH is not set.")
  endif()
endif()

# -------------------- Macros  --------------------------
include(cmake/Macros.cmake)

# -------------------- Library --------------------------
add_subdirectory(runtime)

# -------------------- Applications ---------------------
add_subdirectory(app)
