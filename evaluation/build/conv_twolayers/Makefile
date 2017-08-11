
####################################################
#       Makefile ------- Thomas Chau 2013          #
#       Tested for MaxCompiler 2013.2.2            #
####################################################
# 1)    For SLiC interface only
#       If you are using MaxCompilerRT,
#       use maxfilecompile instead of sliccompile
#       and change LFLAGS -lslic to -lmaxcompilerrt
# 2)    This makefile uses common settings and
#       compilation targets defined in the root folder
# 3)    Usage
#       -----
#       Build and run simulation:  make runsim
#       Kill failed simulation:    make stopsim
#       Debug simulation:          make debug
#       Build DFE:                 make build
#       Run DFE:                   make run
#       Clean binaries:            make clean
#       Clean all generated files: make distclean


# ---- Project settings ----

MAXCOMPILER_BUILD_DIR?=$(HOME)/Desktop/MaxCompiler-Builds
WHEREISROOT=../../..
SRCROOT=../../src

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=ConvTwoLayers
MANAGER=$(PRJ)Manager
APPPKG=conv_twolayers
DEVICE=Maia

BUILD_PARAMS:=bitWidth=$(bitWidth) PF=$(PF) PC=$(PC) PK=$(PK)
BUILD_NAME_OPTION:=_b$(bitWidth)_f$(PF)_c$(PC)_k$(PK)

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
