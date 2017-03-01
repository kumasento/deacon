#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

int main(int argc, char *argv[]) {
  
  max_file_t *maxfile = MaxDeep_init();
  max_engine_t *engine = max_load(maxfile, "local:*");

  const uint64_t K = max_get_constant_uint64t(maxfile, "kernelSize");
  const uint64_t N = max_get_constant_uint64t(maxfile, "numPipes");
  const uint64_t F = max_get_constant_uint64t(maxfile, "freq");

  printf("K = %ld\n", K);
  printf("N = %ld\n", N);
  printf("F = %ld\n", F);

  const int inp_size = 8 * 8 * 3;
  const int wgt_size = 4 * 4 * 3 * 96;
  const int out_size = 5 * 5 * 96;

  uint8_t *data_inp = (uint8_t *) malloc(sizeof(uint32_t) * inp_size);
  uint8_t *data_wgt = (uint8_t *) malloc(sizeof(uint32_t) * wgt_size);
  uint8_t *data_out = (uint8_t *) malloc(sizeof(uint32_t) * out_size);

  // TODO: write stream encoder and decoder methods to transform the
  // data type
  // (2017-02-24) Ruizhe Zhao
  for (int i = 0; i < inp_size * 4; i += 4) {
    uint32_t val = i / 4 + 1;
    for (int j = 0; j < 4; j ++) {
      data_inp[i + j] = (val >> (j * 8));
    }
  }

  for (int i = 0; i < wgt_size * 4; i += 4) {
    uint32_t val = i / 4 + 1;
    for (int j = 0; j < 4; j ++) {
      data_wgt[i + j] = (val >> (j * 8));
    }
  }

  MaxDeep_dramWrite_actions_t inp_write_actions;
  inp_write_actions.param_size_bytes  = inp_size * sizeof(uint32_t);
  inp_write_actions.param_start_bytes = 0;
  inp_write_actions.instream_fromcpu  = data_inp;

  MaxDeep_dramWrite_actions_t wgt_write_actions;
  wgt_write_actions.param_size_bytes  = wgt_size * sizeof(uint32_t);
  wgt_write_actions.param_start_bytes = inp_size * sizeof(uint32_t);
  wgt_write_actions.instream_fromcpu  = data_wgt;

  MaxDeep_dramRead_actions_t out_read_actions;
  out_read_actions.param_size_bytes  = out_size * sizeof(uint32_t);
  out_read_actions.param_start_bytes = (inp_size + wgt_size) * sizeof(uint32_t);
  out_read_actions.outstream_tocpu   = data_out;

  MaxDeep_actions_t actions;
  // actions.param_dataSize = data_size;

  MaxDeep_dramWrite_run(engine, &inp_write_actions);
  MaxDeep_dramWrite_run(engine, &wgt_write_actions);
  MaxDeep_run(engine, &actions);
  MaxDeep_dramRead_run(engine, &out_read_actions);

  for (int i = 0; i < 16; i += 4) {
    uint32_t val = 0;
    for (int j = 0; j < 4; j ++)
      val += (data_out[i + j] << (j * 8));
    printf("out[%4d] = %u\n", i / 4, val);
  }

  return 0;
}
