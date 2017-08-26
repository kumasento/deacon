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
PRJ=ConvLayer
MANAGER=$(PRJ)Manager
APPPKG=conv_single_layer
DEVICE=Maia

H?=32
W?=32
C?=32
F?=32
K?=3
SEQ?=0
USE_DRAM?=false
USE_DRAM_SUFFIX:=PCIe
ifeq ($(USE_DRAM), true)

USE_DRAM_SUFFIX:=DRAM
EXTRA_CFLAGS:=-DUSE_DRAM

endif


BUILD_PARAMS:=bitWidth=$(bitWidth) PF=$(PF) PC=$(PC) PK=$(PK) H=$(H) W=$(W) C=$(C) F=$(F) K=$(K) SEQ=$(SEQ) USE_DRAM=$(USE_DRAM)
BUILD_NAME_OPTION:=_b$(bitWidth)_H$(H)_W$(W)_C$(C)_F$(F)_K$(K)_f$(PF)_c$(PC)_k$(PK)_SEQ$(SEQ)_$(USE_DRAM_SUFFIX)

# Using known hardware settings
include $(WHEREISROOT)/make/Makefile.$(DEVICE).hardware


# Using common compillation targets and compiler settings
include $(WHEREISROOT)/make/Makefile.common
