#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <time.h>

#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

unsigned int burst_aligned_size(unsigned int size, size_t num_bytes, size_t burst_size) {
  return (unsigned int) (ceil((double) size * num_bytes / burst_size) * burst_size / num_bytes);
}

int main(int argc, char *argv[]) {
  printf("\x1B[32mMaxDeep Command Line Program\x1B[0m\n");

  int num_iters = 1000;

  unsigned int conv_height       = 30;
  unsigned int conv_width        = 30;
  unsigned int conv_num_channels = 128;
  unsigned int conv_num_filters  = 128;
  unsigned int conv_kernel_size  = 3;
  unsigned int fc_height         = 16;
  unsigned int fc_width          = 16;
  
  printf("Loading maxfile ...\n");
  max_file_t *maxfile = MaxDeep_init();
  max_engine_t *engine = max_load(maxfile, "local:0");
  // size_t burst_size = max_get_burst_size(maxfile, NULL);
  // printf("Burst size: %ld\n", burst_size);

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
  const uint64_t num_pipes 
    = max_get_constant_uint64t(maxfile, "numPipes");

  printf("Constants:\n");
  printf("\t- maxConvHeight:      %ld\n", max_conv_height);
  printf("\t- maxConvWidth:       %ld\n", max_conv_width);
  printf("\t- maxConvNumChannels: %ld\n", max_conv_num_channels);
  printf("\t- maxConvNumFilters:  %ld\n", max_conv_num_filters);
  printf("\t- maxConvKernelSize:  %ld\n", max_conv_kernel_size);
  printf("\t- maxFCHeight:        %ld\n", max_fc_height);
  printf("\t- maxFCWidth:         %ld\n", max_fc_width);
  printf("\t- numPipes:           %ld\n", num_pipes);

  printf("Computed:\n");
  unsigned int inp_size
    = conv_height * conv_width * conv_num_channels;
  printf("\t- CONV Input size:   %10d\n", inp_size);
  unsigned int wgt_size
    = ( conv_kernel_size
      * conv_kernel_size
      * conv_num_channels
      * ceil((double) conv_num_filters / num_pipes) * num_pipes);
  printf("\t- CONV Weights size: %10d\n", wgt_size);
  unsigned int out_size
    = (conv_height - conv_kernel_size + 1) *
      (conv_width - conv_kernel_size + 1) *
      (ceil((double) conv_num_filters / num_pipes) * num_pipes)
      ;
  printf("\t- CONV Output size:  %10d\n", out_size);
  // unsigned int origin_out_size = out_size;
  unsigned int fc_inp_size = fc_width;
  printf("\t- FC   Input size:   %10d\n", fc_inp_size);
  unsigned int fc_wgt_size = fc_height * fc_width;
  printf("\t- FC   Weights size: %10d\n", fc_wgt_size);
  unsigned int fc_out_size = fc_height;
  printf("\t- FC   Output size:  %10d\n", fc_out_size);

  // printf("\t- CONV Input size:   %10d (%d)\n", inp_size,    burst_aligned_size(inp_size,    sizeof(unsigned int), burst_size));
  // printf("\t- CONV Weights size: %10d (%d)\n", wgt_size,    burst_aligned_size(wgt_size,    sizeof(unsigned int), burst_size));
  // printf("\t- CONV Output size:  %10d (%d)\n", out_size,    burst_aligned_size(out_size,    sizeof(unsigned int), burst_size));
  // printf("\t- FC   Input size:   %10d (%d)\n", fc_inp_size, burst_aligned_size(fc_inp_size, sizeof(unsigned int), burst_size));
  // printf("\t- FC   Weights size: %10d (%d)\n", fc_wgt_size, burst_aligned_size(fc_wgt_size, sizeof(unsigned int), burst_size));
  // printf("\t- FC   Output size:  %10d (%d)\n", fc_out_size, burst_aligned_size(fc_out_size, sizeof(unsigned int), burst_size));

  // inp_size    = burst_aligned_size(inp_size,    sizeof(unsigned int), burst_size);
  // wgt_size    = burst_aligned_size(wgt_size,    sizeof(unsigned int), burst_size);
  // out_size    = burst_aligned_size(out_size,    sizeof(unsigned int), burst_size);
  // fc_inp_size = burst_aligned_size(fc_inp_size, sizeof(unsigned int), burst_size);
  // fc_wgt_size = burst_aligned_size(fc_wgt_size, sizeof(unsigned int), burst_size);
  // fc_out_size = burst_aligned_size(fc_out_size, sizeof(unsigned int), burst_size);

  // uint8_t *data_inp    = (uint8_t *) malloc(sizeof(uint32_t) * inp_size);
  // uint8_t *data_wgt    = (uint8_t *) malloc(sizeof(uint32_t) * wgt_size);
  // uint8_t *data_out    = (uint8_t *) malloc(sizeof(uint32_t) * out_size);
  // uint8_t *data_fc_inp = (uint8_t *) malloc(sizeof(uint32_t) * fc_inp_size);
  // uint8_t *data_fc_wgt = (uint8_t *) malloc(sizeof(uint32_t) * fc_wgt_size);
  // uint8_t *data_fc_out = (uint8_t *) malloc(sizeof(uint32_t) * fc_out_size);
  uint32_t *data_inp    = (uint32_t *) malloc(sizeof(uint32_t) * inp_size);
  uint32_t *data_wgt    = (uint32_t *) malloc(sizeof(uint32_t) * wgt_size);
  uint32_t *data_out    = (uint32_t *) malloc(sizeof(uint32_t) * out_size);
  uint32_t *data_fc_inp = (uint32_t *) malloc(sizeof(uint32_t) * fc_inp_size);
  uint32_t *data_fc_wgt = (uint32_t *) malloc(sizeof(uint32_t) * fc_wgt_size);
  uint32_t *data_fc_out = (uint32_t *) malloc(sizeof(uint32_t) * fc_out_size);
  if (!data_inp || !data_wgt || !data_out || !data_fc_inp || !data_fc_wgt || !data_fc_out)
    fprintf(stderr, "Cannot allocat inp stream\n");
  printf("Allocated memory\n");
  
  // TODO: write stream encoder and decoder methods to transform the
  // data type
  // (2017-02-24) Ruizhe Zhao
  // for (int i = 0; i < (int) inp_size * 4; i += 4) {
  //   uint32_t val = i / 4;
  //   for (int j = 0; j < 4; j ++) {
  //     data_inp[i + j] = (val >> (j * 8));
  //   }
  // }
  // for (int i = 0; i < (int) fc_inp_size * 4; i += 4) {
  //   uint32_t val = i / 4;
  //   for (int j = 0; j < 4; j ++) {
  //     data_fc_inp[i + j] = (val >> (j * 8));
  //   }
  // }
  // for (int i = 0; i < (int) wgt_size * 4; i += 4) {
  //   uint32_t val = i / 4;
  //   for (int j = 0; j < 4; j ++) {
  //     data_wgt[i + j] = (val >> (j * 8));
  //   }
  // }
  // for (int i = 0; i < (int) fc_wgt_size * 4; i += 4) {
  //   uint32_t val = 1;
  //   for (int j = 0; j < 4; j ++) {
  //     data_fc_wgt[i + j] = (val >> (j * 8));
  //   }
  // }
  // uint32_t base_addr = 0;

  for (int i = 0; i < (int) inp_size; i ++) {
    data_inp[i] = (unsigned int) i;
  }
  printf("Assigned data_inp\n");
  for (int i = 0; i < (int) wgt_size; i ++) {
    data_wgt[i] = i;
  }
  printf("Assigned data_wgt\n");
  for (int i = 0; i < (int) fc_inp_size; i ++) {
    data_fc_inp[i] = i;
  }
  for (int i = 0; i < (int) fc_wgt_size; i ++) {
    data_fc_wgt[i] = i;
  }


  // MaxDeep_dramWrite_actions_t inp_write_actions;
  // inp_write_actions.param_size_bytes  = inp_size * sizeof(uint32_t);
  // inp_write_actions.param_start_bytes = base_addr;
  // inp_write_actions.instream_fromcpu  = data_inp;
  // base_addr += inp_size * sizeof(uint32_t);
  // printf("ADDR = 0x%08x\n", base_addr);

  // MaxDeep_dramWrite_actions_t wgt_write_actions;
  // wgt_write_actions.param_size_bytes  = wgt_size * sizeof(uint32_t);
  // wgt_write_actions.param_start_bytes = base_addr;
  // wgt_write_actions.instream_fromcpu  = data_wgt;
  // base_addr += wgt_size * sizeof(uint32_t);
  // printf("ADDR = 0x%08x\n", base_addr);

  // MaxDeep_dramRead_actions_t out_read_actions;
  // out_read_actions.param_size_bytes  = out_size * sizeof(uint32_t);
  // out_read_actions.param_start_bytes = base_addr;
  // out_read_actions.outstream_tocpu   = data_out;
  // base_addr += out_size * sizeof(uint32_t);
  // printf("ADDR = 0x%08x\n", base_addr);

  // MaxDeep_dramWrite_actions_t fc_inp_write_actions;
  // fc_inp_write_actions.param_size_bytes  = fc_inp_size * sizeof(uint32_t);
  // fc_inp_write_actions.param_start_bytes = base_addr;
  // fc_inp_write_actions.instream_fromcpu  = data_fc_inp;
  // base_addr += fc_inp_size * sizeof(uint32_t);
  // printf("ADDR = 0x%08x\n", base_addr);

  // MaxDeep_dramWrite_actions_t fc_wgt_write_actions;
  // fc_wgt_write_actions.param_size_bytes  = fc_wgt_size * sizeof(uint32_t);
  // fc_wgt_write_actions.param_start_bytes = base_addr;
  // fc_wgt_write_actions.instream_fromcpu  = data_fc_wgt;
  // base_addr += fc_wgt_size * sizeof(uint32_t);
  // printf("ADDR = 0x%08x\n", base_addr);

  // MaxDeep_dramRead_actions_t fc_out_read_actions;
  // fc_out_read_actions.param_size_bytes  = fc_out_size * sizeof(uint32_t);
  // fc_out_read_actions.param_start_bytes = base_addr;
  // fc_out_read_actions.outstream_tocpu   = data_fc_out;
  // base_addr += fc_out_size * sizeof(uint32_t);
  // printf("ADDR = 0x%08x\n", base_addr);

  MaxDeep_actions_t actions;
  // max_set_debug((max_actions_t *) &actions, "maxdeep", MAX_DEBUG_ALWAYS);
  actions.param_conv_height       = conv_height;
  actions.param_conv_width        = conv_width;
  actions.param_conv_num_channels = conv_num_channels;
  actions.param_conv_num_filters  = conv_num_filters;
  actions.param_conv_kernel_size  = conv_kernel_size;
  actions.param_conv_num_iters    = 1;
  actions.param_fc_height         = fc_height;
  actions.param_fc_width          = fc_width;
  actions.param_fc_num_iters      = 1;
  actions.instream_conv_cache_inp = data_inp;
  actions.instream_conv_cache_wgt = data_wgt;
  actions.instream_fc_cache_inp   = data_fc_inp;
  actions.instream_fc_cache_wgt   = data_fc_wgt;
  actions.outstream_conv_out      = data_out;
  actions.outstream_fc_out        = data_fc_out;

  // printf("Writing to DRAM ...\n\n");
  // MaxDeep_dramWrite_run(engine, &inp_write_actions);
  // MaxDeep_dramWrite_run(engine, &wgt_write_actions);
  // MaxDeep_dramWrite_run(engine, &fc_inp_write_actions);
  // MaxDeep_dramWrite_run(engine, &fc_wgt_write_actions);

  printf("Computing ...\n\n");

  struct timeval t0, t1;

  gettimeofday(&t0, NULL);
  for (int i = 0; i < num_iters; i ++)
    MaxDeep_run(engine, &actions);
  gettimeofday(&t1, NULL);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
  printf("TOTAL TIME:     %lf s\n", elapsed);
  printf("EACH ITER TIME: %lf s\n", elapsed / num_iters);
  printf("BANDWIDTH TIME: %lf s\n",
      ((double) (inp_size + wgt_size + fc_inp_size + fc_wgt_size) * sizeof(unsigned int)) /
      ((double) 2 * 1024 * 1024 * 1024));

  // printf("Reading back ...\n\n");
  // MaxDeep_dramRead_run(engine, &out_read_actions);
  // MaxDeep_dramRead_run(engine, &fc_out_read_actions);

  // for (int i = 0; i < (int) origin_out_size * 4; i += 4) {
  //   uint32_t val = 0;
  //   for (int j = 0; j < 4; j ++)
  //     val += (data_out[i + j] << (j * 8));
  //   printf("CONV out[%4d] = %u\n", i / 4, val);
  // }
  // for (int i = 0; i < (int) fc_height * 4; i += 4) {
  //   uint32_t val = 0;
  //   for (int j = 0; j < 4; j ++)
  //     val += (data_fc_out[i + j] << (j * 8));
  //   printf("FC   out[%4d] = %u\n", i / 4, val);
  // }

  return 0;
}
