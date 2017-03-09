#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <getopt.h>

#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

unsigned int burst_aligned_size(unsigned int size, size_t num_bytes) {
  return (unsigned int) (ceil((double) size * num_bytes / 384) * 384 / num_bytes);
}

int main(int argc, char *argv[]) {
  printf("\x1B[32mMaxDeep Command Line Program\x1B[0m\n");

  unsigned int conv_height       = 5;
  unsigned int conv_width        = 5;
  unsigned int conv_num_channels = 4;
  unsigned int conv_num_filters  = 4;
  unsigned int conv_kernel_size  = 3;
  unsigned int fc_height         = 16;
  unsigned int fc_width          = 16;
  
  printf("\x1B[32mLoading\x1B[0m maxfile ...\n");
  max_file_t *maxfile = MaxDeep_init();
  max_engine_t *engine = max_load(maxfile, "local:*");

  const uint64_t max_conv_height
    = max_get_constant_uint64t(maxfile, "maxConvHeight");
  const uint64_t max_conv_width
    = max_get_constant_uint64t(maxfile, "maxConvWidth");
  const uint64_t max_conv_num_channels
    = max_get_constant_uint64t(maxfile, "maxConvNumChannels");
  const uint64_t max_conv_num_filters
    = max_get_constant_uint64t(maxfile, "maxConvNumFilters");
  const uint64_t max_conv_kernel_size
    = max_get_constant_uint64t(maxfile, "maxConvKernelSize");
  const uint64_t max_fc_height
    = max_get_constant_uint64t(maxfile, "maxFCHeight");
  const uint64_t max_fc_width
    = max_get_constant_uint64t(maxfile, "maxFCWidth");

  printf("\x1B[32mConstants:\x1B[0m\n");
  printf("\t- maxConvHeight:      %ld\n", max_conv_height);
  printf("\t- maxConvWidth:       %ld\n", max_conv_width);
  printf("\t- maxConvNumChannels: %ld\n", max_conv_num_channels);
  printf("\t- maxConvNumFilters:  %ld\n", max_conv_num_filters);
  printf("\t- maxConvKernelSize:  %ld\n", max_conv_kernel_size);
  printf("\t- maxFCHeight:        %ld\n", max_fc_height);
  printf("\t- maxFCWidth:         %ld\n", max_fc_width);

  unsigned int inp_size
    = conv_height * conv_width * conv_num_channels;
  unsigned int wgt_size
    = conv_kernel_size * conv_kernel_size * conv_num_channels * conv_num_filters;
  unsigned int out_size
    = ((conv_height - conv_kernel_size + 1) *
       (conv_width - conv_kernel_size + 1) *
       conv_num_filters);
  unsigned int fc_inp_size
    = fc_width;
  unsigned int fc_wgt_size
    = fc_height * fc_width;
  unsigned int fc_out_size
    = fc_height;

  printf("\x1B[32mComputed:\x1B[0m\n");
  printf("\t- CONV Input size:   %10d (%d)\n", inp_size, burst_aligned_size(inp_size, sizeof(unsigned int)));
  printf("\t- CONV Weights size: %10d (%d)\n", wgt_size, burst_aligned_size(wgt_size, sizeof(unsigned int)));
  printf("\t- CONV Output size:  %10d (%d)\n", out_size, burst_aligned_size(out_size, sizeof(unsigned int)));
  printf("\t- FC   Input size:   %10d (%d)\n", fc_inp_size, burst_aligned_size(fc_inp_size, sizeof(unsigned int)));
  printf("\t- FC   Weights size: %10d (%d)\n", fc_wgt_size, burst_aligned_size(fc_wgt_size, sizeof(unsigned int)));
  printf("\t- FC   Output size:  %10d (%d)\n", fc_out_size, burst_aligned_size(fc_out_size, sizeof(unsigned int)));

  inp_size    = burst_aligned_size(inp_size, sizeof(unsigned int));
  wgt_size    = burst_aligned_size(wgt_size, sizeof(unsigned int));
  out_size    = burst_aligned_size(out_size, sizeof(unsigned int));
  fc_inp_size = burst_aligned_size(fc_inp_size, sizeof(unsigned int));
  fc_wgt_size = burst_aligned_size(fc_wgt_size, sizeof(unsigned int));
  fc_out_size = burst_aligned_size(fc_out_size, sizeof(unsigned int));

  uint8_t *data_inp    = (uint8_t *) malloc(sizeof(uint32_t) * inp_size);
  uint8_t *data_wgt    = (uint8_t *) malloc(sizeof(uint32_t) * wgt_size);
  uint8_t *data_out    = (uint8_t *) malloc(sizeof(uint32_t) * out_size);
  uint8_t *data_fc_inp = (uint8_t *) malloc(sizeof(uint32_t) * fc_inp_size);
  uint8_t *data_fc_wgt = (uint8_t *) malloc(sizeof(uint32_t) * fc_wgt_size);
  uint8_t *data_fc_out = (uint8_t *) malloc(sizeof(uint32_t) * fc_out_size);

  // TODO: write stream encoder and decoder methods to transform the
  // data type
  // (2017-02-24) Ruizhe Zhao
  for (int i = 0; i < (int) inp_size * 4; i += 4) {
    uint32_t val = 1;
    for (int j = 0; j < 4; j ++) {
      data_inp[i + j] = (val >> (j * 8));
    }
  }
  for (int i = 0; i < (int) fc_inp_size * 4; i += 4) {
    uint32_t val = 1;
    for (int j = 0; j < 4; j ++) {
      data_fc_inp[i + j] = (val >> (j * 8));
    }
  }
  for (int i = 0; i < (int) wgt_size * 4; i += 4) {
    uint32_t val = 1;
    for (int j = 0; j < 4; j ++) {
      data_wgt[i + j] = (val >> (j * 8));
    }
  }
  for (int i = 0; i < (int) fc_wgt_size * 4; i += 4) {
    uint32_t val = 1;
    for (int j = 0; j < 4; j ++) {
      data_fc_wgt[i + j] = (val >> (j * 8));
    }
  }

  uint32_t base_addr = 0;

  MaxDeep_dramWrite_actions_t inp_write_actions;
  inp_write_actions.param_size_bytes  = inp_size * sizeof(uint32_t);
  inp_write_actions.param_start_bytes = base_addr;
  inp_write_actions.instream_fromcpu  = data_inp;
  base_addr += inp_size * sizeof(uint32_t);
  printf("ADDR = 0x%08x\n", base_addr);

  MaxDeep_dramWrite_actions_t wgt_write_actions;
  wgt_write_actions.param_size_bytes  = wgt_size * sizeof(uint32_t);
  wgt_write_actions.param_start_bytes = base_addr;
  wgt_write_actions.instream_fromcpu  = data_wgt;
  base_addr += wgt_size * sizeof(uint32_t);
  printf("ADDR = 0x%08x\n", base_addr);

  MaxDeep_dramRead_actions_t out_read_actions;
  out_read_actions.param_size_bytes  = out_size * sizeof(uint32_t);
  out_read_actions.param_start_bytes = base_addr;
  out_read_actions.outstream_tocpu   = data_out;
  base_addr += out_size * sizeof(uint32_t);
  printf("ADDR = 0x%08x\n", base_addr);

  MaxDeep_dramWrite_actions_t fc_inp_write_actions;
  fc_inp_write_actions.param_size_bytes  = fc_inp_size * sizeof(uint32_t);
  fc_inp_write_actions.param_start_bytes = base_addr;
  fc_inp_write_actions.instream_fromcpu  = data_fc_inp;
  base_addr += fc_inp_size * sizeof(uint32_t);
  printf("ADDR = 0x%08x\n", base_addr);

  MaxDeep_dramWrite_actions_t fc_wgt_write_actions;
  fc_wgt_write_actions.param_size_bytes  = fc_wgt_size * sizeof(uint32_t);
  fc_wgt_write_actions.param_start_bytes = base_addr;
  fc_wgt_write_actions.instream_fromcpu  = data_fc_wgt;
  base_addr += fc_wgt_size * sizeof(uint32_t);
  printf("ADDR = 0x%08x\n", base_addr);

  MaxDeep_dramRead_actions_t fc_out_read_actions;
  fc_out_read_actions.param_size_bytes  = fc_out_size * sizeof(uint32_t);
  fc_out_read_actions.param_start_bytes = base_addr;
  fc_out_read_actions.outstream_tocpu   = data_fc_out;
  base_addr += fc_out_size * sizeof(uint32_t);
  printf("ADDR = 0x%08x\n", base_addr);

  MaxDeep_actions_t actions;
  actions.param_conv_height       = conv_height;
  actions.param_conv_width        = conv_width;
  actions.param_conv_num_channels = conv_num_channels;
  actions.param_conv_num_filters  = conv_num_filters;
  actions.param_conv_kernel_size  = conv_kernel_size;
  actions.param_fc_height         = fc_height;
  actions.param_fc_width          = fc_width;

  MaxDeep_dramWrite_run(engine, &inp_write_actions);
  MaxDeep_dramWrite_run(engine, &wgt_write_actions);
  MaxDeep_dramWrite_run(engine, &fc_inp_write_actions);
  MaxDeep_dramWrite_run(engine, &fc_wgt_write_actions);
  MaxDeep_run(engine, &actions);
  MaxDeep_dramRead_run(engine, &out_read_actions);
  MaxDeep_dramRead_run(engine, &fc_out_read_actions);

  for (int i = 0; i < (int) ((conv_height - conv_kernel_size + 1) * (conv_width - conv_kernel_size + 1) * conv_num_filters * 4); i += 4) {
    uint32_t val = 0;
    for (int j = 0; j < 4; j ++)
      val += (data_out[i + j] << (j * 8));
    printf("CONV out[%4d] = %u\n", i / 4, val);
  }
  for (int i = 0; i < (int) fc_height * 4; i += 4) {
    uint32_t val = 0;
    for (int j = 0; j < 4; j ++)
      val += (data_fc_out[i + j] << (j * 8));
    printf("FC   out[%4d] = %u\n", i / 4, val);
  }

  return 0;
}