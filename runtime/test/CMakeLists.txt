enable_testing()

add_executable(test_layers ${CMAKE_CURRENT_LIST_DIR}/test_layers.cc)
target_include_directories(test_layers
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)
add_test(test_layers test_layers)
