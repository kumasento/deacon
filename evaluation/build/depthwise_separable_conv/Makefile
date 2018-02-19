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
PRJ=DepthwiseSeparableConvLayer
MANAGER=$(PRJ)Manager
APPPKG=depthwise_separable_conv
DEVICE=Maia

BW?=16
P_F?=1
P_K?=1
P_C?=1
H?=4
W?=4
C?=16
F?=4
K?=3
SEQ?=0
FREQ?=100
VERSION?=2
NUM_ITERS?=1
BATCH_SIZE?=1

# ----- USE_DRAM or NOT parameters -----
# USE_DRAM?=true
# USE_DRAM_SUFFIX:=DRAM
# EXTRA_CFLAGS:=-DUSE_DRAM

USE_DRAM?=false
USE_DRAM_SUFFIX:=
EXTRA_CFLAGS:=

ifeq ($(USE_DRAM), false)

USE_DRAM_SUFFIX:=PCIe
EXTRA_CFLAGS:=

endif

EXTRA_CFLAGS+=-DDEPTHWISE_SEPARABLE_V2 -DBIT_WIDTH=$(BW)

BUILD_PARAMS:=bitWidth=$(BW) PF=$(P_F) PK=$(P_K) PC=$(P_C) H=$(H) W=$(W) C=$(C) F=$(F) K=$(K) SEQ=$(SEQ) USE_DRAM=$(USE_DRAM) FREQ=$(FREQ) VERSION=$(VERSION)
BUILD_NAME_OPTION:=_b$(BW)_H$(H)_W$(W)_C$(C)_F$(F)_K$(K)_f$(P_F)_c$(P_C)_k$(P_K)_SEQ$(SEQ)_$(USE_DRAM_SUFFIX)_FREQ_$(FREQ)_V$(VERSION)

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
