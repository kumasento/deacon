
add_subdirectory (proto)
add_subdirectory (hardware)
add_subdirectory (interface)

add_executable (maxdeep ${PROJECT_SOURCE_DIR}/src/maxdeep.cpp)
target_link_libraries (maxdeep interface)
