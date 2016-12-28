# external projects for gflags and glog
include (ExternalProject)
# pthread {{{
find_package (Threads REQUIRED)
add_library(pthread INTERFACE)
target_link_libraries(pthread INTERFACE ${CMAKE_THREAD_LIBS_INIT})
# }}}
# gflags {{{
set (GFLAGS_INSTALL "${CMAKE_BINARY_DIR}/external/gflags-install")
set (GFLAGS_PREFIX "${CMAKE_BINARY_DIR}/external/gflags-prefix")
ExternalProject_Add (
  gflags
  URL "https://github.com/gflags/gflags/archive/master.zip"
  PREFIX ${GFLAGS_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL}
)
set (GFLAGS_INCLUDE_DIRS ${GFLAGS_INSTALL}/include)
set (GFLAGS_LIBRARIES ${GFLAGS_INSTALL}/lib/libgflags.a)
set (GFLAGS_LIBRARY_DIRS ${GFLAGS_INSTALL}/lib)

add_library (libgflags IMPORTED GLOBAL STATIC)
add_dependencies (libgflags gflags)
set_target_properties (libgflags PROPERTIES "IMPORTED_LOCATION" "${GFLAGS_LIBRARIES}")
include_directories (${GFLAGS_INCLUDE_DIRS})
# }}}
# glogs {{{
set (GLOG_INSTALL "${CMAKE_BINARY_DIR}/external/glog-install")
set (GLOG_PREFIX "${CMAKE_BINARY_DIR}/external/glog-prefix")
ExternalProject_Add(
  glog
  DEPENDS gflags
  URL "https://github.com/google/glog/archive/master.zip"
  PREFIX ${GLOG_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL}
  CONFIGURE_COMMAND env ${GLOG_PREFIX}/src/glog/configure --prefix=${GLOG_INSTALL} --with-gflags=${GFLAGS_LIBRARY_DIRS}/..  --enable-shared=no
)
set(GLOG_INCLUDE_DIRS ${GLOG_INSTALL}/include)
set(GLOG_LIBRARIES ${GLOG_INSTALL}/lib/libglog.a)
set(GLOG_LIBRARY_DIRS ${GLOG_INSTALL}/lib)

add_library(libglog IMPORTED GLOBAL STATIC)
add_dependencies(libglog glog)
set_target_properties(libglog PROPERTIES "IMPORTED_LOCATION" "${GLOG_LIBRARIES}")
include_directories(${GLOG_INCLUDE_DIRS})
# }}}
