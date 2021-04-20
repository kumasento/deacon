macro(add_deacon_app name)
  set(oneValueArgs DFEMODEL)
  set(multiValueArgs CPUCODE KERNEL MANAGER)
  cmake_parse_arguments(ARG
    ""
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN})
  # DFE build.
  # CPU executable target.
  add_executable("${name}" "${ARG_CPUCODE}")
  # Run simulation
  # Run dfe
endmacro()
