# Check the build directory
if (EXISTS $ENV{HOME}/.MaxCompiler_build_user.conf)
  set (MAXCOMPILER_BUILD_DIR /mnt/data/scratch/$ENV{USER}/builds)
else ()
  message (STATUS "Cannot find MAXCOMPILER_BUILD_DIR, using the default")
  set (MAXCOMPILER_BUILD_DIR ${PROJECT_BINARY_DIR}/maxeler)
endif ()
message (STATUS "MAXCOMPILER_BUILD_DIR: ${MAXCOMPILER_BUILD_DIR}")

# Configure MaxCompiler related paths
if (DEFINED ENV{MAXCOMPILERDIR} AND DEFINED ENV{MAXELEROSDIR})
  message (STATUS "Maxeler environments are all set")
  message (STATUS "MAXCOMPILER:  $ENV{MAXCOMPILERDIR}")
  message (STATUS "MAXELEROSDIR: $ENV{MAXELEROSDIR}")

  include_directories ("$ENV{MAXCOMPILERDIR}/include")
  include_directories ("$ENV{MAXCOMPILERDIR}/include/slic")
  include_directories ("$ENV{MAXELEROSDIR}/include")

  link_directories ("$ENV{MAXCOMPILERDIR}/lib")
  link_directories ("$ENV{MAXELEROSDIR}/lib")

  set (MAXCOMPILERDIR $ENV{MAXCOMPILERDIR} CACHE PATH
    "MaxJ compiler directory")
  set (MAXELEROSDIR $ENV{MAXELEROSDIR} CACHE PATH
    "MaxJ OS runtime directory")
endif ()

# Add a .max file target
function (add_max prj dfemodel enginefiles jars deps)
  set (SIMMAXNAME ${prj}_${dfemodel}_DFE_SIM) 
  set (DFEMAXNAME ${prj}_${dfemodel}_DFE) 
  set (SIMMAXDIR ${MAXCOMPILER_BUILD_DIR}/${SIMMAXNAME}/results)
  set (DFEMAXDIR ${MAXCOMPILER_BUILD_DIR}/${DFEMAXNAME}/results)

  # Flags for maxjc
  set (MAXJC maxjc)
  set (EXTRA_JARS ${jars})
  set (JFLAGS -cp ${MAXCOMPILERDIR}/lib/MaxCompiler.jar:${EXTRA_JARS} -1.6 -d .)

  set (ENGINEFILES ${enginefiles})
  set (MAXAPPNAME ${prj})
  set (MAXFILENAME ${MAXAPPNAME}.max)
  set (MAXFILE ${SIMMAXDIR}/${MAXFILENAME})

  message (STATUS "MAXFILE: ${MAXFILE}")
  message (STATUS "depends: ${deps}")

  string (TOLOWER ${prj} MAXPKG)
  set (MAXJAVARUN maxJavaRun)
  set (MAXJAVARUNMEMSIZE 8192)
  set (MAXMPCX "false")
  add_custom_command (
    OUTPUT ${MAXFILE}
    # Generate Java Classes
    COMMAND ${MAXJC} ${JFLAGS} ${ENGINEFILES}
    # Generate MaxFile
    COMMAND MAXAPPJCP=${EXTRA_JARS}:.
      MAXAPPPKG=${MAXPKG}
      ${MAXJAVARUN} -m ${MAXJAVARUNMEMSIZE} ${MAXAPPNAME}
      DFEModel=${dfemodel}
      maxFileName=${MAXAPPNAME}
      target='DFE_SIM'
      enableMPCX=${MAXMPCX}
    COMMAND 
    DEPENDS ${deps} ${ENGINEFILES}
  )
  add_custom_target (maxfile DEPENDS ${MAXFILE})

  set (SLICCOMPILE sliccompile)
  add_custom_command (
    OUTPUT ${MAXAPPNAME}_sim.o
    COMMAND ${SLICCOMPILE} ${MAXFILE} ${MAXAPPNAME}_sim.o
    DEPENDS ${MAXFILE}
  )
  set_source_files_properties (
    ${MAXAPPNAME}_sim.o
    PROPERTIES EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  add_library (maxfile_sim ${MAXAPPNAME}_sim.o)
  set_target_properties (maxfile_sim PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories (maxfile_sim PUBLIC
    ${SIMMAXDIR})
endfunction ()

# add a PHONY runsim target, exe should be specified and has its own 
# generate command
function (add_runsim exe dfemodel)
  # MaxJ simulation system configuration parameters
  set (MAXCOMPILERSIM maxcompilersim)
  set (MAXJ_DFE_MODEL ${dfemodel})
  set (MAXJ_SIM_SYSTEM_NAME "$ENV{USER}a")
  set (MAXJ_SIM_SYSTEM_ID "$ENV{USER}a0")
  set (MAXJ_SIM_SYSTEM_NUM_DEVICES "1")
  set (MAXJ_SIM_SYSTEM_LD_PRELOAD "$ENV{MAXCOMPILERDIR}/lib/maxeleros-sim/lib/libmaxeleros.so")
  set (MAXJ_SLIC_CONF "use_simulation=${MAXJ_SIM_SYSTEM_NAME}")
  set (MAXJ_SIM_EXECUTABLE ${exe})

  # Run simulation, no file generated
  add_custom_target(
    runsim
    # Restart the simulation system
    COMMAND ${MAXCOMPILERSIM}
      -n ${MAXJ_SIM_SYSTEM_NAME}
      -c ${MAXJ_DFE_MODEL}
      -d ${MAXJ_SIM_SYSTEM_NUM_DEVICES} restart
    # Execute the simulation program
    COMMAND SLIC_CONF=${MAXJ_SLIC_CONF}
      LD_PRELOAD=${MAXJ_SIM_SYSTEM_LD_PRELOAD} 
      ${PROJECT_BINARY_DIR}/${MAXJ_SIM_EXECUTABLE}
      ${MAXJ_SIM_SYSTEM_ID}:${MAXJ_SIM_SYSTEM_NAME}
    DEPENDS ${MAXJ_SIM_EXECUTABLE}
  )

endfunction ()
