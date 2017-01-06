file (GLOB_RECURSE ENGINEFILES ${PROJECT_SOURCE_DIR}/src/*.maxj)
file (GLOB_RECURSE INTERFACEFILES ${PROJECT_SOURCE_DIR}/src/interface/*.cpp)
file (GLOB TEST_SRCS ${PROJECT_SOURCE_DIR}/src/test/*.cpp)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/src/test)

# add_runsim_test {{{
function (add_runsim_test test_name)
  set (TEST_NAME ${test_name})
  set (TEST_SRC ${TEST_SRC_DIR}/${TEST_NAME}.cpp)

  message (STATUS "Adding test: ${TEST_NAME}")

  # create executable target for the test
  add_executable (${TEST_NAME} ${TEST_SRC})
  add_executable (${TEST_NAME}_dfe ${TEST_SRC})
  
  # add dependency maxfile
  set (MAXJ_PROJECT_NAME "MaxDeep")
  set (MAXJ_DFE_MODEL "MAX3424A")
  set (MAXJ_DEPEND_JARS
    ${PROJECT_BINARY_DIR}/java_gen/maxdeep.pb.jar:${PROJECT_SOURCE_DIR}/third_party/protobuf-java-3.1.0.jar)
  set (MAXJ_DEPEND_TARGETS
    protojava
    ${PROJECT_BINARY_DIR}/java_gen/maxdeep.pb.jar
    ${TEST_SRC_DIR}/${TEST_NAME}.prototxt)
  set (MAXJ_BUILD_PARAMS
    "protoFile=${TEST_SRC_DIR}/${TEST_NAME}.prototxt")

  # after this function has been applied, a MAXFILE_TARGET_NAME based 
  # targets will be generated and can be referenced in the current scope
  add_max(${TEST_NAME}
    ${MAXJ_PROJECT_NAME}
    ${MAXJ_DFE_MODEL}
    ${MAXJ_BUILD_PARAMS}
    "${ENGINEFILES}"
    ${MAXJ_DEPEND_JARS}
    "${MAXJ_DEPEND_TARGETS}"
    "SIM")

  message (STATUS "Test target: ${MAXFILE_TARGET_NAME}")

  # create interface target, which includes the third party libraries
  # and other custom libraries
  set (INTERFACE_TARGET_NAME interface_${TEST_NAME})
  add_library (${INTERFACE_TARGET_NAME} ${INTERFACEFILES})
  include_directories (${INTERFACE_TARGET_NAME} ${PROJECT_SOURCE_DIR}/include)
  add_dependencies (${INTERFACE_TARGET_NAME}
    ${MAXFILE_TARGET_NAME}_sim
    libglog
    libgflags
    libgtest)
  target_link_libraries (${INTERFACE_TARGET_NAME}
    ${MAXFILE_TARGET_NAME}_sim
    protocpp
    pthread
    libglog
    libgflags
    libgtest)
  # link to the executable file for this test
  target_link_libraries (${TEST_NAME} ${INTERFACE_TARGET_NAME})

  add_max(${TEST_NAME}
    ${MAXJ_PROJECT_NAME}
    ${MAXJ_DFE_MODEL}
    ${MAXJ_BUILD_PARAMS}
    "${ENGINEFILES}"
    ${MAXJ_DEPEND_JARS}
    "${MAXJ_DEPEND_TARGETS}"
    "DFE")
  add_library (${INTERFACE_TARGET_NAME}_dfe ${INTERFACEFILES})
  include_directories (${INTERFACE_TARGET_NAME}_dfe ${PROJECT_SOURCE_DIR}/include)
  add_dependencies (${INTERFACE_TARGET_NAME}_dfe
    ${MAXFILE_TARGET_NAME}_dfe
    libglog
    libgflags
    libgtest)
  target_link_libraries (${INTERFACE_TARGET_NAME}_dfe
    ${MAXFILE_TARGET_NAME}_dfe
    protocpp
    pthread
    libglog
    libgflags
    libgtest)
  target_link_libraries (${TEST_NAME}_dfe ${INTERFACE_TARGET_NAME}_dfe)

  # create runsim command for this test
  add_runsim (${TEST_NAME} ${MAXJ_DFE_MODEL})

  message (STATUS "runsim target: ${RUNSIM_TARGET_NAME}")

  # finally create the test
  add_test (
    NAME ${TEST_NAME}
    COMMAND make ${RUNSIM_TARGET_NAME}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  add_rundfe(${TEST_NAME}_dfe ${MAXJ_DFE_MODEL})
endfunction ()
# }}}

add_runsim_test (test_conv_layer)
add_runsim_test (test_fc_layer)
add_runsim_test (test_pooling_layer)
add_runsim_test (test_relu_layer)
add_runsim_test (test_lenet)
