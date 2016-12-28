file (GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/src/interface/*.cpp)

add_library(interface ${srcs})
add_dependencies(interface libglog libgflags maxfile_sim)
# protocpp is added in proto/
target_link_libraries(interface maxfile_sim protocpp libglog libgflags pthread)
