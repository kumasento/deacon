file(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cc)

# Look for GoogleTest
find_package(GTest REQUIRED)

enable_testing()

add_executable (run_tests ${SRCS})
target_link_libraries(run_tests PUBLIC
  ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES} pthread ${GLOG_LIBRARIES})
target_include_directories(run_tests PUBLIC
  ${GMOCK_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIR}
  "${CMAKE_SOURCE_DIR}/runtime/include")
