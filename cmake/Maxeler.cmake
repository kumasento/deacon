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
function (add_max target prj dfemodel params enginefiles jars deps maxtarget)
  set (MAXJAVACLASS ${prj})
  set (MAXAPPNAME ${target}_${prj})
  set (SIMMAXNAME ${MAXAPPNAME}_${dfemodel}_DFE_SIM) 
  set (DFEMAXNAME ${MAXAPPNAME}_${dfemodel}_DFE) 
  set (SIMMAXDIR ${MAXCOMPILER_BUILD_DIR}/${SIMMAXNAME}/results)
  set (DFEMAXDIR ${MAXCOMPILER_BUILD_DIR}/${DFEMAXNAME}/results)

  # Flags for maxjc
  set (MAXJC maxjc)
  set (EXTRA_JARS ${jars})
  set (JFLAGS -cp ${MAXCOMPILERDIR}/lib/MaxCompiler.jar:${EXTRA_JARS} -1.6 -d .)

  set (ENGINEFILES ${enginefiles})
  set (MAXFILENAME ${MAXAPPNAME}.max)
  if (${maxtarget} STREQUAL "SIM")
    set (MAXFILE ${SIMMAXDIR}/${MAXFILENAME})
    set (MAXTARGET "DFE_SIM")
  else ()
    set (MAXFILE ${DFEMAXDIR}/${MAXFILENAME})
    set (MAXTARGET "DFE")
  endif ()

  message (STATUS "MAXFILE: ${MAXFILE}")

  if (${maxtarget} STREQUAL "SIM")
    set (SLICSUFFIX "_sim")
  else ()
    set (SLICSUFFIX "_dfe")
  endif ()

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
      MAXSOURCEDIRS=${PROJECT_SOURCE_DIR}/src/hardware
      ${MAXJAVARUN} -m ${MAXJAVARUNMEMSIZE} ${MAXJAVACLASS}
      DFEModel=${dfemodel}
      maxFileName=${MAXAPPNAME}
      target=${MAXTARGET}
      enableMPCX=${MAXMPCX}
      ${params}
    DEPENDS ${deps} ${ENGINEFILES}
  )
  set (MAXFILE_TARGET_NAME maxfile_${prj}_${dfemodel}_${target}${SLICSUFFIX})
  set (MAXFILE_TARGET_NAME maxfile_${prj}_${dfemodel}_${target}${SLICSUFFIX} PARENT_SCOPE)
  add_custom_target (${MAXFILE_TARGET_NAME} DEPENDS ${MAXFILE})

  set (SLICCOMPILE sliccompile)

  message (STATUS "SLIC SUFFIX: ${SLICSUFFIX}")

  add_custom_command (
    OUTPUT ${MAXAPPNAME}${SLICSUFFIX}.o
    COMMAND ${SLICCOMPILE} ${MAXFILE} ${MAXAPPNAME}${SLICSUFFIX}.o
    DEPENDS ${MAXFILE}
  )
  set_source_files_properties (
    ${MAXAPPNAME}${SLICSUFFIX}.o
    PROPERTIES EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  add_library (${MAXFILE_TARGET_NAME}${SLICSUFFIX} ${MAXAPPNAME}${SLICSUFFIX}.o)
  target_include_directories (
    ${MAXFILE_TARGET_NAME}${SLICSUFFIX}
    PUBLIC
    ${MAXCOMPILERDIR}/include
    ${MAXCOMPILERDIR}/include/slic
    ${MAXELEROSDIR}/include
    )

  link_directories (${MAXCOMPILERDIR}/lib ${MAXELEROSDIR}/lib)
  target_link_libraries (
    ${MAXFILE_TARGET_NAME}${SLICSUFFIX}
    maxeleros
    slic
    m
    pthread
    )

  set_target_properties (${MAXFILE_TARGET_NAME}${SLICSUFFIX} PROPERTIES LINKER_LANGUAGE CXX)
  if (${maxtarget} STREQUAL "SIM")
    target_include_directories (${MAXFILE_TARGET_NAME}${SLICSUFFIX} PUBLIC
      ${SIMMAXDIR})
  else ()
    target_include_directories (${MAXFILE_TARGET_NAME}${SLICSUFFIX} PUBLIC
      ${DFEMAXDIR})
  endif ()
endfunction (add_max)

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

  set (RUNSIM_TARGET_NAME runsim_${exe})
  set (RUNSIM_TARGET_NAME runsim_${exe} PARENT_SCOPE)
  # Run simulation, no file generated
  add_custom_target(
    ${RUNSIM_TARGET_NAME}
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
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )

endfunction ()

function (add_rundfe exe dfemodel)
  set (MAXJ_DFE_SYSTEM_LD_PRELOAD "/opt/maxeler/maxeleros/lib/libmaxeleros.so")
  set (RUNDFE_TARGET_NAME rundfe_${exe})
  set (RUNDFE_TARGET_NAME rundfe_${exe} PARENT_SCOPE)
  add_custom_target(
    ${RUNDFE_TARGET_NAME}
    COMMAND LD_PRELOAD=${MAXJ_DFE_SYSTEM_LD_PRELOAD}
      ${PROJECT_BINARY_DIR}/${exe}
    DEPENDS ${exe}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )
endfunction ()
