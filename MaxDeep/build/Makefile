
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

WHEREISROOT=../..

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=MaxDeep
APPPKG=maxdeep
KERNEL=$(PRJ)Kernel
MANAGER=$(PRJ)Manager

NUM_PIPES?=1
FREQ?=100
MULTI_PUMPING_FACTOR?=1
MAX_CONV_KERNEL_SIZE?=3
MAX_CONV_HEIGHT?=32
MAX_CONV_WIDTH?=32
MAX_CONV_NUM_CHANNELS?=512
MAX_CONV_NUM_FILTERS?=512
DEBUG?=0

BUILD_PARAMS:=\
	numPipes=$(NUM_PIPES)\
	freq=$(FREQ)\
	multiPumpingFactor=$(MULTI_PUMPING_FACTOR)\
	maxConvKernelSize=$(MAX_CONV_KERNEL_SIZE)\
	maxConvHeight=$(MAX_CONV_HEIGHT)\
	maxConvWidth=$(MAX_CONV_WIDTH)\
	maxConvNumChannels=$(MAX_CONV_NUM_CHANNELS)\
	maxConvNumFilters=$(MAX_CONV_NUM_FILTERS)\
	dbg=$(DEBUG)

BUILD_NAME_OPTION:=_N_$(NUM_PIPES)_H_$(MAX_CONV_HEIGHT)_W_$(MAX_CONV_WIDTH)_C_$(MAX_CONV_NUM_CHANNELS)_F_$(MAX_CONV_NUM_FILTERS)_K_$(MAX_CONV_KERNEL_SIZE)_C_$(MULTI_PUMPING_FACTOR)_F_$(FREQ)MHz

# Using known hardware settings
include $(WHEREISROOT)/Makefile.Maia.hardware

# Using common compillation targets and compiler settings
include $(WHEREISROOT)/Makefile.common

# Using utilities Makefile
# include Makefile.utils
