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

void run_loopback_test(max_file_t *maxfile, max_engine_t *engine);

int main(int argc, char *argv[]) {
  printf("\x1B[33mMaxDeep Command Line Program\x1B[0m\n");

  printf("\x1B[32mLOADING\x1B[0m maxfile ...\n");
  max_file_t *maxfile = MaxDeep_init();
  max_engine_t *engine = max_load(maxfile, "*:0");

  run_loopback_test(maxfile, engine);

  return 0;
}

/**
 * This test is aiming to find out the bandwidth and power
 * consumption of DRAM read and write on the Maxeler platform.
 *
 * @author Ruizhe Zhao
 * @since 16/05/2017
 * @param maxfile max_file_t pointer
 * @param engine max_engine_t *engine
 */
void run_loopback_test(max_file_t *maxfile, max_engine_t *engine) {
  printf("\x1B[32mTESTING\x1B[0m loopback ...\n");
  const int num_iters = 1;

  size_t burst_size = max_get_burst_size(maxfile, NULL);
  printf("Burst size: %ld\n", burst_size);

  const int64_t inp_size = burst_size * 1024 * 1024;
  const int64_t wgt_size = inp_size;
  const int64_t out_size = inp_size;

  uint32_t *inp = (uint32_t *) malloc(sizeof(uint32_t) * inp_size);
  uint32_t *wgt = (uint32_t *) malloc(sizeof(uint32_t) * wgt_size);
  uint32_t *out = (uint32_t *) malloc(sizeof(uint32_t) * out_size);
  uint32_t *expected_out = (uint32_t *) malloc(sizeof(uint32_t) * out_size);
  if (!inp || !wgt || !out) {
    fprintf(stderr, "Cannot allocat streams\n");
    exit(-1);
  }

  for (int i = 0; i < (int) inp_size; i++)
    inp[i] = (uint32_t) i;
  for (int i = 0; i < (int) wgt_size; i++)
    wgt[i] = (uint32_t) i;
  for (int i = 0; i < (int) out_size; i++)
    expected_out[i] = (uint32_t) i + i;

  MaxDeep_dramWrite_actions_t inp_write_actions;
  inp_write_actions.param_size_bytes = inp_size * sizeof(uint32_t);
  inp_write_actions.param_start_bytes = 0;
  inp_write_actions.instream_fromcpu = (const uint8_t *) inp;

  MaxDeep_dramWrite_actions_t wgt_write_actions;
  wgt_write_actions.param_size_bytes = wgt_size * sizeof(uint32_t);
  wgt_write_actions.param_start_bytes = inp_size * sizeof(uint32_t);
  wgt_write_actions.instream_fromcpu = (const uint8_t *) wgt;

  MaxDeep_dramRead_actions_t read_actions;
  read_actions.param_size_bytes = out_size * sizeof(uint32_t);
  read_actions.param_start_bytes = (inp_size + wgt_size) * sizeof(uint32_t);
  read_actions.outstream_tocpu = (uint8_t *) out;

  MaxDeep_actions_t actions;
  actions.param_num_elems = inp_size;

  printf("Writing to DRAM ...\n\n");
  MaxDeep_dramWrite_run(engine, &inp_write_actions);
  MaxDeep_dramWrite_run(engine, &wgt_write_actions);

  printf("Computing ...\n\n");

  struct timeval t0, t1;

  gettimeofday(&t0, NULL);
  for (int i = 0; i < num_iters; i ++)
    MaxDeep_run(engine, &actions);
  gettimeofday(&t1, NULL);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
  elapsed /= num_iters;

  printf("RUN TIME:       %lf s\n", elapsed);
  printf("FREQUENCY:      %.2f MHz\n", inp_size / elapsed * 1e-6);

  printf("Reading back ...\n\n");
  MaxDeep_dramRead_run(engine, &read_actions);
  for (int i = 0; i < 10; i ++)
    printf("out[%3d] = %u\n", i, out[i]);
  for (int i = 0; i < out_size; i ++) {
    if (out[i] != expected_out[i]) {
      fprintf(stderr,
          "out[%3d] = %u is different from expected result %u\n",
          i, out[i], expected_out[i]);
    }
  }
  printf("TEST PASSED if no error shows up\n");

  free(inp);
  free(out);
}
