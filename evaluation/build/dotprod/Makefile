
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

MAXCOMPILER_BUILD_DIR:=$(NAS_HOME)/maxcompiler_builds
WHEREISROOT=../../..
SRCROOT=$(WHEREISROOT)/evaluation/src

DEVICE?=Maia

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=DotProd
APPPKG=dotprod

ifeq ($(DEVICE),Lima)
	MANAGER=$(PRJ)Max5Manager
else
	MANAGER=$(PRJ)Manager
endif

BUILD_PARAMS:=bitWidth=$(BIT_WIDTH) vecSize=$(VEC_SIZE)
BUILD_NAME_OPTION:=_b$(BIT_WIDTH)_n$(VEC_SIZE)

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
