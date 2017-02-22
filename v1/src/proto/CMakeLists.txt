# Protobuf: find the environment variables at first
# note that this requires the CMake and the Protobuf are installed under 
# a same path prefix.
find_package (Protobuf REQUIRED)
# Generate protobuf code for Java and C++
include_directories (${PROTOBUF_INCLUDE_DIR})

# Check whether protoc compiler exists
if (EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message (STATUS "Found protobuf compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else ()
  message (FATAL_ERROR "Cannot find protobuf compiler")
endif ()

# Setup several variables to denote:
# PROTO_FILE: The path to the **single** source .proto file
# PROTO_CPP_GEN_DIR: The generate directory of the C++ file
# PROTO_CPP_FILE_NAME: The name of the generated C++ file
set (PROTO_PATH "${PROJECT_SOURCE_DIR}/src/proto")
set (PROTO_FILE "maxdeep.proto")

# Generate CPP interface {{{
set (PROTO_CPP_GEN_DIR "${PROJECT_BINARY_DIR}/cpp_gen")
set (PROTO_CPP_SRC_NAME "maxdeep.pb.cc")
set (PROTO_CPP_HDR_NAME "maxdeep.pb.h")
set (PROTO_CPP_GEN_FILES 
  ${PROTO_CPP_GEN_DIR}/${PROTO_CPP_SRC_NAME}
  ${PROTO_CPP_GEN_DIR}/${PROTO_CPP_HDR_NAME})

# Include the generated directory
include_directories (${PROTO_CPP_GEN_DIR})

# The following command will generate the source files from protoc
# which is listed in the official CMake tutorial (they suggest to 
# add the code generation target as a custom command).
add_custom_command (
  OUTPUT "${PROTO_CPP_GEN_DIR}/${PROTO_CPP_SRC_NAME}"
         "${PROTO_CPP_GEN_DIR}/${PROTO_CPP_HDR_NAME}"
  # create the generate directory at first, it will not be generated 
  # automatically by protoc
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PROTO_CPP_GEN_DIR}"
  # the protoc command, which follows the standard usage
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    --cpp_out=${PROTO_CPP_GEN_DIR}
    -I${PROTO_PATH}
    ${PROTO_PATH}/${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  COMMENT "Running protoc on ${PROTO_FILE} to generate CPP code" VERBATIM
)

# Let CMake know that these files will be generated
set_source_files_properties(${PROTO_CPP_GEN_FILES} PROPERTIES GENERATED TRUE)
add_library(protocpp STATIC ${PROTO_CPP_GEN_FILES})

# link the include directories into the protocpp target
# which will allow those targets linked to protocpp to include these 
# directories
target_include_directories (protocpp PUBLIC ${PROTO_CPP_GEN_DIR})
target_include_directories (protocpp PUBLIC ${PROTOBUF_INCLUDE_DIR})
# }}}
# Generate Java Interface {{{
# protojava will be used to generate the target .max file
find_package (Java REQUIRED)
include (UseJava)

set (PROTO_JAVA_GEN_DIR ${PROJECT_BINARY_DIR}/java_gen)
set (PROTO_JAVA_SRC_DIR ${PROTO_JAVA_GEN_DIR}/maxdeep/proto)
set (PROTO_JAVA_SRC_NAME "MaxDeepProtos.java")
set (PROTO_JAVA_SRC_FILE ${PROTO_JAVA_SRC_DIR}/${PROTO_JAVA_SRC_NAME})

add_custom_command (
  OUTPUT ${PROTO_JAVA_SRC_DIR}/${PROTO_JAVA_SRC_NAME}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROTO_JAVA_GEN_DIR}
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    --java_out=${PROTO_JAVA_GEN_DIR}
    -I${PROTO_PATH}
    ${PROTO_PATH}/${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  COMMENT "Running protoc on ${PROTO_FILE} to generate Java code" VERBATIM
)

set (PROTO_JAVA_JAR ${PROJECT_SOURCE_DIR}/third_party/protobuf-java-3.1.0.jar)
set_source_files_properties(${PROTO_JAVA_SRC_FILE} PROPERTIES GENERATED TRUE)
add_custom_command(
  OUTPUT ${PROTO_JAVA_GEN_DIR}/maxdeep.pb.jar
  COMMAND javac -d ${PROTO_JAVA_GEN_DIR} ${PROTO_JAVA_SRC_FILE} -cp ${PROTO_JAVA_JAR}
  COMMAND jar cf ${PROTO_JAVA_GEN_DIR}/maxdeep.pb.jar -C ${PROTO_JAVA_GEN_DIR} .
  DEPENDS ${PROTO_JAVA_SRC_FILE}
  COMMENT "Generating maxdeep.pb.jar"
)
add_custom_target(protojava DEPENDS ${PROTO_JAVA_GEN_DIR}/maxdeep.pb.jar)
# }}}
