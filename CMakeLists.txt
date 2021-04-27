cmake_minimum_required(VERSION 3.10)

project(deacon LANGUAGES C CXX)

# C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(MAXCOMPILER_PATH "" CACHE PATH "Path to the root of MaxCompiler.")
set(MAXCOMPILER_BUILD_DIR "${CMAKE_BINARY_DIR}" CACHE PATH "Where MaxCompiler build results should be.")

# -------------------- Configuration --------------------
if (EXISTS "${MAXCOMPILER_PATH}")
  # Java 1.8 is required if we're using MaxCompiler.
  find_package(Java 1.8 REQUIRED)
  include(UseJava) # Load the Java plugin.

  message(STATUS "Using MaxCompiler: ${MAXCOMPILER_PATH}")
  message(STATUS "MaxCompiler builds will be placed at: ${MAXCOMPILER_BUILD_DIR}")
  
  # MaxCompiler executables.
  find_program(MAXJC maxjc PATHS "${MAXCOMPILER_PATH}/bin" REQUIRED "Path to maxjc.")
  find_program(MAXJAVARUN maxJavaRun PATHS "${MAXCOMPILER_PATH}/bin" REQUIRED "Path to maxJavaRun.")

  # JAR files.
  find_jar(MAXCOMPILERJAR MaxCompiler PATHS "${MAXCOMPILER_PATH}/lib"
            REQUIRED "Path to MaxCompiler.jar.")
  find_jar(MAX4PLATFORMJAR Max4Platform PATHS "${MAXCOMPILER_PATH}/lib"
            REQUIRED "Path to Max4Platform.jar.")
  find_jar(MAX5PLATFORMJAR Max5Platform PATHS "${MAXCOMPILER_PATH}/lib"
            REQUIRED "Path to Max5Platform.jar.")
endif()

# -------------------- Macros  --------------------------
include(cmake/Macros.cmake)

# -------------------- Library --------------------------
add_subdirectory(runtime)

# -------------------- Applications ---------------------
add_subdirectory(app)
