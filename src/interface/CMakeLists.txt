file (GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/src/interface/*.cpp)

add_library(interface ${srcs})
target_link_libraries(interface protocpp)
