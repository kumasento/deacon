
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

WHEREISROOT=../../..

# Assuming Kernel and Manager class names follow the pattern below:
PRJ=MaxDeep
APPPKG=maxdeep
KERNEL=$(PRJ)Kernel
MANAGER=$(PRJ)Manager

NUM_PIPES?=1
FREQ?=100
KERNEL_SIZE?=3
MULTI_PUMPING_FACTOR?=1
MAX_CONV_KERNEL_SIZE?=3
MAX_CONV_HEIGHT?=32
MAX_CONV_WIDTH?=32
MAX_CONV_NUM_CHANNELS?=512
MAX_CONV_NUM_FILTERS?=512
MAX_FC_HEIGHT?=4096
MAX_FC_WIDTH?=4096
NUM_FC_ROW_PIPES?=1
NUM_FC_COL_PIPES?=1
NUM_CONV_CHNL_PIPES?=1
NUM_CONV_FLTR_PIPES?=1
NUM_MPDP_FACTOR?=1
DEBUG?=0
DESIGN_NAME?=LOOPBACK
ONE_DIM_CONV_WINDOW_WIDTH?=1
BITWIDTH?=32
DSPFACTOR?=1

BUILD_PARAMS:=\
	bitWidth=$(BITWIDTH)\
	dspFactor=$(DSPFACTOR)\
	designName=$(DESIGN_NAME)\
	numPipes=$(NUM_PIPES)\
	freq=$(FREQ)\
	kernelSize=$(KERNEL_SIZE)\
	multiPumpingFactor=$(MULTI_PUMPING_FACTOR)\
	maxConvKernelSize=$(MAX_CONV_KERNEL_SIZE)\
	maxConvHeight=$(MAX_CONV_HEIGHT)\
	maxConvWidth=$(MAX_CONV_WIDTH)\
	maxConvNumChannels=$(MAX_CONV_NUM_CHANNELS)\
	maxConvNumFilters=$(MAX_CONV_NUM_FILTERS)\
	maxFCHeight=$(MAX_FC_HEIGHT)\
	maxFCWidth=$(MAX_FC_WIDTH)\
	numFCRowPipes=$(NUM_FC_ROW_PIPES)\
	numFCColPipes=$(NUM_FC_COL_PIPES)\
	numConvChnlPipes=$(NUM_CONV_CHNL_PIPES)\
	numConvFltrPipes=$(NUM_CONV_FLTR_PIPES)\
	numMPDPFactor=$(NUM_MPDP_FACTOR)\
	oneDimConvWindowWidth=$(ONE_DIM_CONV_WINDOW_WIDTH)\
	dbg=$(DEBUG)

#BUILD_NAME_OPTION:=#_N_$(NUM_PIPES)_H_$(MAX_CONV_HEIGHT)_W_$(MAX_CONV_WIDTH)_C_$(MAX_CONV_NUM_CHANNELS)_F_$(MAX_CONV_NUM_FILTERS)_K_$(MAX_CONV_KERNEL_SIZE)_FH_$(MAX_FC_HEIGHT)_FW_$(MAX_FC_WIDTH)_FR_$(NUM_FC_ROW_PIPES)_FC_$(NUM_FC_COL_PIPES)_MPDP_$(NUM_MPDP_FACTOR)_C_$(MULTI_PUMPING_FACTOR)_F_$(FREQ)MHz
BUILD_NAME_OPTION:=_$(DESIGN_NAME)
ifeq ($(DESIGN_NAME),MULT_ARRAY)
	BUILD_NAME_OPTION:=$(BUILD_NAME_OPTION)_P_$(NUM_PIPES)
else ifeq ($(DESIGN_NAME),ONE_DIM_CONV)
	BUILD_NAME_OPTION:=$(BUILD_NAME_OPTION)_W_$(ONE_DIM_CONV_WINDOW_WIDTH)_P_$(NUM_PIPES)_M_$(MULTI_PUMPING_FACTOR)
else ifeq ($(DESIGN_NAME),CONV2D)
	BUILD_NAME_OPTION:=$(BUILD_NAME_OPTION)_P_$(NUM_CONV_CHNL_PIPES)_$(NUM_CONV_FLTR_PIPES)_M_$(MULTI_PUMPING_FACTOR)_K_$(KERNEL_SIZE)
	BUILD_NAME_OPTION:=$(BUILD_NAME_OPTION)_$(MAX_CONV_HEIGHT)_$(MAX_CONV_WIDTH)_$(MAX_CONV_NUM_CHANNELS)_$(MAX_CONV_NUM_FILTERS)
	BUILD_NAME_OPTION:=$(BUILD_NAME_OPTION)_B_$(BITWIDTH)_DSP_$(DSPFACTOR)
endif
BUILD_NAME_OPTION:=$(BUILD_NAME_OPTION)_FREQ_$(FREQ)MHz

# Using known hardware settings
# include $(WHEREISROOT)/Makefile.Max3.hardware
include $(WHEREISROOT)/Makefile.Maia.hardware

# Using common compillation targets and compiler settings
include $(WHEREISROOT)/Makefile.common

# Using utilities Makefile
# include Makefile.utils
