file (GLOB_RECURSE ENGINEFILES ${PROJECT_SOURCE_DIR}/src/*.maxj)
# Setup MaxJ build system
set (MAXJ_PROJECT_NAME "MaxDeep")
set (MAXJ_DFE_MODEL "MAX3424A")
set (MAXJ_DEPEND_JARS
  ${PROJECT_BINARY_DIR}/java_gen/maxdeep.pb.jar:${PROJECT_SOURCE_DIR}/third_party/protobuf-java-3.1.0.jar)
# Only depends on both the jar file (trigger update) and the target protojava
# (let CMake find the build command) will make it work correctly
set (MAXJ_DEPEND_TARGETS protojava ${PROJECT_BINARY_DIR}/java_gen/maxdeep.pb.jar)

add_max(${MAXJ_PROJECT_NAME} ${MAXJ_DFE_MODEL} "${ENGINEFILES}"
  ${MAXJ_DEPEND_JARS} "${MAXJ_DEPEND_TARGETS}")
