file (GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/src/interface/*.cpp)


add_library(interface ${srcs})
add_dependencies(interface libglog libgflags)
# protocpp is added in proto/
target_link_libraries(interface protocpp libglog libgflags pthread)
