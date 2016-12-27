# Protobuf
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
set (PROTO_FILE "${PROJECT_SOURCE_DIR}/src/proto/maxdeep.proto")
set (PROTO_CPP_GEN_DIR "${PROJECT_BINARY_DIR}/include/maxdeep/proto")
set (PROTO_CPP_FILE_NAME "maxdeep")

# Include the generated directory
include_directories ("${PROJECT_BINARY_DIR}/include")

# Set generate target
add_custom_command (
  OUTPUT "${PROTO_CPP_GEN_DIR}/${PROTO_CPP_FILE_NAME}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PROTO_CPP_GEN_DIR}"
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out ${PROTO_CPP_GEN_DIR}
    ${PROTO_CPP_FILE_NAME}
  DEPENDS ${PROTO_FILE}
  COMMENT "Running protoc on ${PROTO_FILE}" VERBATIM
)
