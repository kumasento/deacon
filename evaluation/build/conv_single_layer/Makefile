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
PRJ=ConvSingleLayer
MANAGER=$(PRJ)Manager
APPPKG=conv_single_layer
DEVICE=Maia

bitWidth?=16
DTYPE?=fixed
H?=32
W?=32
C?=32
F?=32
K?=3
P_F?=1
P_C?=1
P_K?=1
SEQ?=0
FREQ?=100
USE_DRAM?=true
USE_DRAM_SUFFIX:=PCIe
USE_WINO?=false
USE_WINO_SUFFIX:=
WINO_COEFF_OFFLINE?=false
WINO_COEFF_OFFLINE_SUFFIX:=
NUM_COEFF_FIFO_SPLITS?=1
NUM_COEFF_FIFO_SPLITS_SUFFIX:=
DEBUG?=false
DEBUG_SUFFIX:=
TRACE?=false

EXTRA_CFLAGS:=

ifeq ($(USE_DRAM), true)
USE_DRAM_SUFFIX:=DRAM
EXTRA_CFLAGS+= -DUSE_DRAM
endif

ifeq ($(USE_WINO), true)
USE_WINO_SUFFIX:=_WINO
EXTRA_CFLAGS+= -DUSE_WINO
endif

ifeq ($(WINO_COEFF_OFFLINE), true)
WINO_COEFF_OFFLINE_SUFFIX:=_COEF
EXTRA_CFLAGS+= -DWINO_COEFF_OFFLINE
endif

ifeq ($(TRACE), true)
EXTRA_CFLAGS+= -DTRACE
endif

ifeq ($(DEBUG), true)
DEBUG_SUFFIX:=_DEBUG
endif

ifneq ($(NUM_COEFF_FIFO_SPLITS), 1)
NUM_COEFF_FIFO_SPLITS_SUFFIX:=_S$(NUM_COEFF_FIFO_SPLITS)
endif

BUILD_PARAMS:=bitWidth=$(bitWidth) PF=$(P_F) PC=$(P_C) PK=$(P_K) H=$(H) W=$(W) C=$(C) F=$(F) K=$(K) SEQ=$(SEQ) USE_DRAM=$(USE_DRAM) FREQ=$(FREQ) DEBUG=$(DEBUG) USE_WINOGRAD=$(USE_WINO) WINOGRAD_WEIGHTS_OFFLINE=$(WINO_COEFF_OFFLINE) DTYPE=$(DTYPE) NUM_COEFF_FIFO_SPLITS=$(NUM_COEFF_FIFO_SPLITS)
BUILD_NAME_OPTION:=_b$(bitWidth)_H$(H)_W$(W)_C$(C)_F$(F)_K$(K)_f$(P_F)_c$(P_C)_k$(P_K)_SEQ$(SEQ)_$(USE_DRAM_SUFFIX)_FREQ_$(FREQ)$(USE_WINO_SUFFIX)$(WINO_COEFF_OFFLINE_SUFFIX)$(NUM_COEFF_FIFO_SPLITS_SUFFIX)$(DEBUG_SUFFIX)

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
