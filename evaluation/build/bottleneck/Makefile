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

# MAXCOMPILER_BUILD_DIR?=/mnt/ccnas2/bdp/rz3515/maxcompiler_builds
WHEREISROOT=../../..
SRCROOT=../../src

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=Bottleneck
MANAGER=$(PRJ)Manager
APPPKG=bottleneck
DEVICE=Lima


BIT_WIDTH?=8
WBW?=2
H?=7
W?=7
C?=2048
F?=512
K?=3
S?=1
P_F?=1
P_C?=1
P_K?=1
SEQ0?=0
SEQ1?=0
SEQ2?=0
FREQ?=200
USE_DRAM?=true
USE_DRAM_SUFFIX:=DRAM
COEFF_ON_CHIP?=true
COEFF_ON_CHIP_SUFFIX:=_COC
EXTRA_CFLAGS:=-DUSE_DRAM
DEBUG:=false

ifeq ($(DEBUG), true)
EXTRA_CFLAGS+=-DTRACE
DEBUG_SUFFIX:=_DBG
endif

ifeq ($(USE_DRAM), false)
USE_DRAM_SUFFIX:=PCIe
EXTRA_CFLAGS:=
endif

ifeq ($(COEFF_ON_CHIP), false)
COEFF_ON_CHIP_SUFFIX:=
endif

EXTRA_CFLAGS+=-DBIT_WIDTH=$(BIT_WIDTH) -DWBW=$(WBW)

BUILD_PARAMS:=bitWidth=$(BIT_WIDTH) WBW=$(WBW) PF=$(P_F) PC=$(P_C) PK=$(P_K) H=$(H) W=$(W) C=$(C) F=$(F) K=$(K) SEQ0=$(SEQ0) SEQ1=$(SEQ1) SEQ2=$(SEQ2) USE_DRAM=$(USE_DRAM) FREQ=$(FREQ) COEFF_ON_CHIP=$(COEFF_ON_CHIP) DEBUG=$(DEBUG) INIT_COEFF=true
BUILD_NAME_OPTION:=_b$(BIT_WIDTH)_$(WBW)_H$(H)_W$(W)_C$(C)_F$(F)_K$(K)_S$(S)_f$(P_F)_c$(P_C)_k$(P_K)_SEQ$(SEQ0)_$(SEQ1)_$(SEQ2)_$(USE_DRAM_SUFFIX)$(COEFF_ON_CHIP_SUFFIX)_FREQ_$(FREQ)$(DEBUG_SUFFIX)

SIMDEVICEID?=b
CLI_OPTIONS?=-n 2 

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
