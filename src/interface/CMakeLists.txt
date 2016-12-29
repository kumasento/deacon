file (GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/src/interface/*.cpp)

add_library(interface ${srcs})
set (MAXFILE_TARGET_NAME_SIM maxfile_MaxDeep_MAX3424A_main_sim)
add_dependencies(interface libglog libgflags ${MAXFILE_TARGET_NAME_SIM})
# protocpp is added in proto/
target_link_libraries(interface
  ${MAXFILE_TARGET_NAME_SIM} protocpp libglog libgflags pthread)
