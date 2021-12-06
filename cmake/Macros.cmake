# Create an application built upon Deacon.
#
# It will add the following targets, given the application `name`:
#
#    runsim-${name}             -- build and run design in SIM.
#
macro(add_deacon_app name)
  set(oneValueArgs DFEMODEL MANAGER)
  set(multiValueArgs CPUCODE KERNELS)
  cmake_parse_arguments(ARG
    ""
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN})
  # ----- Local variables
  # Where the class files will locate.
  set(maxcompiler_output_dir "${CMAKE_CURRENT_BINARY_DIR}/bin")
  # Extract manager class name from the manager file.
  get_filename_component(manager_class_name "${ARG_MANAGER}" NAME_WE)
  # Get the pkg name and the maxfile name.
  string(REPLACE "-" "_" pkgname "${name}")
  set(sim_maxfile_name "${pkgname}_sim")
  # DFE build.

  # Get all the source files in their absolute paths.
  # First add the manager file, then all the kernel files.
  list(APPEND src_files "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_MANAGER}")
  foreach(kernel_file ${ARG_KERNELS})
    list(APPEND src_files "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file}") 
  endforeach()

  # Set the simulated design as a build target. The output "sim-${pkgname}" is a 
  # dummy file that won't be generated: it just serves as the indicator of the
  # dependencies between the following command/target.
  add_custom_command(
    OUTPUT "sim-${pkgname}"
    COMMAND maxjc -d "${maxcompiler_output_dir}" -1.8 -nowarn -cp "${MAXCOMPILERJAR}:${MAX4PLATFORMJAR}:${MAX5PLATFORMJAR}" ${src_files}
    COMMAND CLASSPATH=${maxcompiler_output_dir} MAXAPPJCP=${MAX5PLATFORMJAR} MAXAPPPKG="${pkgname}" MAXSOURCEDIRS="${CMAKE_CURRENT_SOURCE_DIR}" maxJavaRun -v -m 8192 "${manager_class_name}" target='DFE_SIM' maxFileName="${sim_maxfile_name}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  add_custom_target("build-sim-${name}"
    DEPENDS "sim-${pkgname}"
    COMMAND cp "$(cat "${CMAKE_CURRENT_BINARY_DIR}/.maxdc_builds_${pkgname}_sim")"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")  


  # CPU executable target.
  # add_executable("${name}" "${ARG_CPUCODE}")

  # Add custom commands like runsim-${name}
  # Run simulation
  # Add custom commands like rundfe-${name}
  # Run dfe
endmacro()
