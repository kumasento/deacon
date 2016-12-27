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
