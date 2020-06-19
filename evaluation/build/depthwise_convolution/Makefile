
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


# MAXCOMPILER_BUILD_DIR:=$(HOME)/Desktop/MaxCompiler-Builds
WHEREISROOT=../../..
SRCROOT=../../src

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=DepthwiseConvolution
MANAGER=$(PRJ)Manager
APPPKG=depthwise_convolution
DEVICE=Lima

TILE_WIDTH?=32
TILE_HEIGHT?=32
TILE_DEPTH?=32
KERNEL_SIZE?=3
PAR_WIDTH?=1
PAR_DEPTH?=1
FREQ?=200
DEBUG?=false
USE_WINOGRAD?=false
WINOGRAD?=0
DEBUG_STR?=

ifeq (${DEBUG},true)
	DEBUG_STR=_debug
endif
ifeq (${USE_WINOGRAD},true)
	WINOGRAD=1
endif

BUILD_PARAMS:=TILE_HEIGHT=${TILE_HEIGHT} TILE_WIDTH=${TILE_WIDTH} TILE_DEPTH=${TILE_DEPTH} PAR_WIDTH=${PAR_WIDTH} PAR_DEPTH=${PAR_DEPTH} FREQ=${FREQ} KERNEL_SIZE=${KERNEL_SIZE} DEBUG=${DEBUG} USE_WINOGRAD=${USE_WINOGRAD}
BUILD_NAME_OPTION:=_th${TILE_HEIGHT}_tw${TILE_WIDTH}_tc${TILE_DEPTH}_pw${PAR_WIDTH}_pc${PAR_DEPTH}_wg${WINOGRAD}_${FREQ}MHz${DEBUG_STR}

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
