cmake_minimum_required(VERSION 2.8.0)
project(maxdeep)

macro(use_cxx11)
if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
endif ()
endmacro(use_cxx11)

use_cxx11()

include(runtime/CMakeLists.txt)
