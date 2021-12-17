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
DEVICE=Lima

BW?=16
WBW?=16
NUM_FRAC_BITS?=8
P_F?=1
P_K?=1
P_C?=1
H?=4
W?=4
C?=16
F?=4
K?=3
S?=1
SEQ?=0
FREQ?=100
VERSION?=1
NUM_ITERS?=1
BATCH_SIZE?=1
EXTRA_CFLAGS:=

# ----- USE_DRAM or NOT parameters -----
# USE_DRAM?=true
# USE_DRAM_SUFFIX:=DRAM
# EXTRA_CFLAGS:=-DUSE_DRAM

USE_DRAM?=true
USE_DRAM_SUFFIX:=DRAM
EXTRA_CFLAGS:=-DUSE_DRAM

COEFF_ON_CHIP?=true
COEFF_ON_CHIP_SUFFIX:=_COC

DEBUG?=false
DEBUG_SUFFIX:=

ifeq ($(USE_DRAM), false)
USE_DRAM_SUFFIX:=PCIe
EXTRA_CFLAGS:=
endif

ifeq ($(COEFF_ON_CHIP), false)
COEFF_ON_CHIP_SUFFIX:=
endif

ifeq ($(DEBUG), true)
DEBUG_SUFFIX:=_DBG
EXTRA_CFLAGS+=-DTRACE
endif

COEFF_FILE?=

NUM_LAYER?=1

EXTRA_CFLAGS+=-DBIT_WIDTH=$(BW) -DWBW=$(WBW)
EXTRA_CFLAGS+=-DDEPTHWISE_SEPARABLE_V2
EXTRA_CFLAGS+=-DNUM_LAYER=$(NUM_LAYER)

BUILD_PARAMS:=bitWidth=$(BW) WBW=$(WBW) NUM_LAYER=$(NUM_LAYER) COEFF_ON_CHIP=$(COEFF_ON_CHIP) PF=$(P_F) PK=$(P_K) PC=$(P_C) H=$(H) W=$(W) C=$(C) F=$(F) K=$(K) SEQ=$(SEQ) USE_DRAM=$(USE_DRAM) S=$(S) PAD=$(P) FREQ=$(FREQ) VERSION=$(VERSION) DEBUG=$(DEBUG) COEFF_FILE=$(COEFF_FILE) NUM_FRAC_BITS=$(NUM_FRAC_BITS)
BUILD_NAME_OPTION:=_b$(BW)_$(WBW)_$(NUM_FRAC_BITS)_N$(NUM_LAYER)_H$(H)_W$(W)_C$(C)_F$(F)_K$(K)_S$(S)_f$(P_F)_c$(P_C)_k$(P_K)_SEQ$(SEQ)_$(USE_DRAM_SUFFIX)_FREQ_$(FREQ)_V$(VERSION)$(DEBUG_SUFFIX)

CLI_OPTIONS?=-n 2

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
