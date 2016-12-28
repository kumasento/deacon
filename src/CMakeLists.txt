add_subdirectory (proto)
add_subdirectory (hardware)
add_subdirectory (interface)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable (maxdeep ${PROJECT_SOURCE_DIR}/src/maxdeep.cpp)
target_link_libraries (maxdeep interface)

add_runsim (maxdeep "MAX3424A")
