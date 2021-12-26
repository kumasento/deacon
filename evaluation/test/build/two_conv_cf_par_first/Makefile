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
WHEREISROOT=/mnt/ccnas2/bdp/rz3515/projects/deacon
SRCROOT=../../src

PRJ=TwoConvCfParFirst
MANAGER=$(PRJ)Manager
APPPKG=two_conv_cf_par_first
DEVICE=Lima



EXTRA_CFLAGS:=

COEFF_FILE?=
# Allow tweaking the debug configuration.
DEBUG?=false
DEBUG_SUFFIX:=
ifeq ($(DEBUG), true)
DEBUG_SUFFIX:=_DBG
EXTRA_CFLAGS+=-DTRACE
endif


FREQ?=200

BUILD_PARAMS:= FREQ=$(FREQ) DEBUG=$(DEBUG) COEFF_FILE=$(COEFF_FILE) 
BUILD_NAME_OPTION:=_FREQ_$(FREQ)$(DEBUG_SUFFIX)
CLI_OPTIONS?=-n 2
SIMDEVICEID?=a

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware
# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
