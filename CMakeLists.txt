cmake_minimum_required(VERSION 3.1)

# --------------------------- Project setting
project(maxdeep C CXX)

# --------------------------- Utilities
include(cmake/Macros.cmake)

# --------------------------- C/C++ setting
# Setup C++11
use_cxx11()

# include(cmake/FindGlog.cmake)

# include(runtime/CMakeLists.txt)

add_subdirectory(pyra)