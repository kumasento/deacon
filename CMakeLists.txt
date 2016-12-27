cmake_minimum_required(VERSION 2.6)

project("maxdeep")

# enable C++11 features
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

message (STATUS "Binary directory: ${PROJECT_BINARY_DIR}")
message (STATUS "Source directory: ${PROJECT_SOURCE_DIR}")

add_subdirectory(src/)
