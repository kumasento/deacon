file (GLOB_RECURSE MAXJ_SOURCES ${PROJECT_SOURCE_DIR}/src/*.maxj)
# Setup MaxJ build system
set (MAXJ_PROJECT_NAME "maxdeep")
set (MAXJ_DFE_MODEL "MAX3424A")

# MaxJ simulation system configuration parameters
set (MAXCOMPILERSIM maxcompilersim)
set (MAXJ_SIM_SYSTEM_NAME "$ENV{USER}a")
set (MAXJ_SIM_SYSTEM_ID "$ENV{USER}a0")
set (MAXJ_SIM_SYSTEM_NUM_DEVICES 1)
set (MAXJ_SIM_SYSTEM_LD_PRELOAD "$ENV{MAXELEROSDIR_SIM}/lib/libmaxeleros.so")
set (MAXJ_SLIC_CONF "use_simulation=${MAXJ_SIM_SYSTEM_NAME}")
set (MAXJ_SIM_EXECUTABLE "${PROJECT_BINARY_DIR}/${MAXJ_PROJECT_NAME}_sim")

# Run simulation, no file generated
add_custom_target(
  runsim
  # Restart the simulation system
  COMMAND ${MAXCOMPILERSIM} -n ${MAXJ_SIM_SYSTEM_NAME} -c ${MAXJ_DFE_MODEL} -d ${MAXJ_SIM_NUM_DEVICES} restart
  # Execute the simulation program
  COMMAND SLIC_CONF=${MAXJ_SLIC_CONF} LD_PRELOAD=${MAXJ_SIM_SYSTEM_LD_PRELOAD} ${MAXJ_SIM_EXECUTABLE} ${MAXJ_SIM_SYSTEM_ID}:${MAXJ_SIM_SYSTEM_NAME}
  DEPENDS ${MAXJ_SIM_EXECUTABLE}
)

# Build the simulation program
add_custom_command(
  OUTPUT ${MAXJ_SIM_EXECUTABLE}.o
  DEPENDS ${MAXJ_SOURCES}
)

# Add run targets {{{
if (DEFINED ENV{MAXCOMPILERDIR} AND DEFINED ENV{MAXELEROSDIR})
  message             (STATUS "Maxeler environments are all set")
  message             (STATUS "MAXCOMPILER:  $ENV{MAXCOMPILERDIR}")
  message             (STATUS "MAXELEROSDIR: $ENV{MAXELEROSDIR}")
  include_directories ("$ENV{MAXCOMPILERDIR}/include")
  include_directories ("$ENV{MAXCOMPILERDIR}/include/slic")
  include_directories ("$ENV{MAXELEROSDIR}/include")
  link_directories    ("$ENV{MAXCOMPILERDIR}/lib")
  link_directories    ("$ENV{MAXELEROSDIR}/lib")
endif ()
# }}}
