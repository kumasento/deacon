
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

MAXCOMPILER_BUILD_DIR:=$(HOME)/Desktop/MaxCompiler-Builds
WHEREISROOT=../../..
SRCROOT=../../src

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=PointwiseConvolution
MANAGER=$(PRJ)Manager
APPPKG=pointwise_convolution
DEVICE=Maia

TILE_WIDTH?=32
TILE_HEIGHT?=32
TILE_IN_DEPTH?=32
TILE_OUT_DEPTH?=32
PAR_WIDTH?=1
PAR_IN_DEPTH?=1
PAR_OUT_DEPTH?=1
FREQ?=200
DEBUG?=false
DEBUG_STR?=

ifeq (${DEBUG},true)
	DEBUG_STR=_debug
endif

BUILD_PARAMS:=TILE_HEIGHT=${TILE_HEIGHT} TILE_WIDTH=${TILE_WIDTH} TILE_IN_DEPTH=${TILE_IN_DEPTH} TILE_OUT_DEPTH=${TILE_OUT_DEPTH} PAR_WIDTH=${PAR_WIDTH} PAR_IN_DEPTH=${PAR_IN_DEPTH} PAR_OUT_DEPTH=${PAR_OUT_DEPTH} FREQ=${FREQ} DEBUG=${DEBUG}
BUILD_NAME_OPTION:=_th${TILE_HEIGHT}_tw${TILE_WIDTH}_tc${TILE_IN_DEPTH}_tf${TILE_OUT_DEPTH}_pw${PAR_WIDTH}_pc${PAR_IN_DEPTH}_pf${PAR_OUT_DEPTH}_${FREQ}MHz${DEBUG_STR}

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
