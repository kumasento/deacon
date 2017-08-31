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

# ------ Constants ----------
COMMA:=,

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=LeNet
MANAGER=$(PRJ)Manager
APPPKG=lenet
DEVICE=Maia

BW?=8
PP?=1,1,1
PPP?=$(subst $(COMMA),_,$(PP))
USE_DEPTH?=false
USE_DEPTH_SUFFIX?=STD
FREQ?=100

ifeq ($(USE_DEPTH),true)
	EXTRA_CFLAGS:=-DUSE_DEPTH
	USE_DEPTH_SUFFIX:=DWS
endif


BUILD_PARAMS:=BW=$(BW) PP=$(PP) FREQ=$(FREQ) USE_DEPTH=$(USE_DEPTH)
BUILD_NAME_OPTION:=_b$(BW)_PP$(PPP)_$(USE_DEPTH_SUFFIX)_FREQ_$(FREQ)

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
