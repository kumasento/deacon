/**
 * This header file contains several test runners for the MaxDeep project.
 */
#ifndef MAXDEEP_TEST_RUNNER_H__
#define MAXDEEP_TEST_RUNNER_H__

#include "maxdeep/utils.h"

namespace maxdeep {

namespace test_runners {

/**
 * This test is aiming to find out the bandwidth and power
 * consumption of DRAM read and write on the Maxeler platform.
 *
 * @author Ruizhe Zhao
 * @since 16/05/2017
 *
 * @param is_sim whether the current environment is simulation
 * @param maxfile max_file_t pointer
 * @param engine max_engine_t *engine
 */
void run_loopback_test(bool is_sim, max_file_t *maxfile, max_engine_t *engine) {
  printf("\x1B[32mTESTING\x1B[0m LOOPBACK ...\n");

  const int num_iters = 1;
  size_t burst_size = max_get_burst_size(maxfile, NULL);

  int64_t inp_size = 1000;
  // Increase the total size if not in the simulation environment
  if (!is_sim)
    inp_size = inp_size * 1024 * 1024;

  int64_t wgt_size = inp_size;
  int64_t out_size = inp_size;

  int64_t burst_aligned_inp_size = utils::burst_aligned_size(inp_size, sizeof(uint32_t), burst_size);
  int64_t burst_aligned_wgt_size = burst_aligned_inp_size;
  int64_t burst_aligned_out_size = burst_aligned_inp_size;

  uint32_t *inp = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_inp_size);
  uint32_t *wgt = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_wgt_size);
  uint32_t *out = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_out_size);
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
  inp_write_actions.param_size_bytes = burst_aligned_inp_size * sizeof(uint32_t);
  inp_write_actions.param_start_bytes = 0;
  inp_write_actions.instream_fromcpu = (const uint8_t *) inp;

  MaxDeep_dramWrite_actions_t wgt_write_actions;
  wgt_write_actions.param_size_bytes = burst_aligned_wgt_size * sizeof(uint32_t);
  wgt_write_actions.param_start_bytes = burst_aligned_inp_size * sizeof(uint32_t);
  wgt_write_actions.instream_fromcpu = (const uint8_t *) wgt;

  MaxDeep_dramRead_actions_t read_actions;
  read_actions.param_size_bytes = burst_aligned_out_size * sizeof(uint32_t);
  read_actions.param_start_bytes = (burst_aligned_inp_size + burst_aligned_wgt_size) * sizeof(uint32_t);
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
  free(wgt);
  free(out);
}

/**
 * Run MULT_ARRAY test
 *
 * @author Ruizhe Zhao
 * @since 19/05/2017
 */
bool run_mult_array_test(bool is_sim, max_file_t *maxfile, max_engine_t *engine) {
  printf("\x1B[32mTESTING\x1B[0m loopback ...\n");

  const int num_iters = 1;
  size_t burst_size = max_get_burst_size(maxfile, NULL);
  int num_pipes = (int) max_get_constant_uint64t(maxfile, "NUM_PIPES");

  int64_t inp_size = 10000;
  // Increase the total size if not in the simulation environment
  if (!is_sim)
    inp_size = inp_size * 1024 * 1024;

  int64_t wgt_size = inp_size * num_pipes;
  int64_t out_size = inp_size;

  int64_t burst_aligned_inp_size = utils::burst_aligned_size(inp_size, sizeof(uint32_t), burst_size);
  int64_t burst_aligned_wgt_size = utils::burst_aligned_size(wgt_size, sizeof(uint32_t), burst_size);
  int64_t burst_aligned_out_size = burst_aligned_inp_size;

  uint32_t *inp = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_inp_size);
  uint32_t *wgt = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_wgt_size);
  uint32_t *out = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_out_size);

  uint32_t *expected_out = (uint32_t *) malloc(sizeof(uint32_t) * out_size);
  if (!inp || !wgt || !out) {
    fprintf(stderr, "Cannot allocat streams\n");
    exit(-1);
  }

  for (int i = 0; i < (int) inp_size; i++)
    inp[i] = (uint32_t) i;
  for (int i = 0; i < (int) wgt_size; i++)
    wgt[i] = (uint32_t) i;
  for (int i = 0; i < (int) out_size; i++) {
    uint32_t sum = 0;
    uint32_t inp_val = inp[i];
    for (int j = i * num_pipes; j < (i + 1) * num_pipes; j ++)
      sum += inp_val * wgt[j];
    expected_out[i] = sum;
  }

  MaxDeep_dramWrite_actions_t inp_write_actions;
  inp_write_actions.param_size_bytes = burst_aligned_inp_size * sizeof(uint32_t);
  inp_write_actions.param_start_bytes = 0;
  inp_write_actions.instream_fromcpu = (const uint8_t *) inp;

  MaxDeep_dramWrite_actions_t wgt_write_actions;
  wgt_write_actions.param_size_bytes = burst_aligned_wgt_size * sizeof(uint32_t);
  wgt_write_actions.param_start_bytes = burst_aligned_inp_size * sizeof(uint32_t);
  wgt_write_actions.instream_fromcpu = (const uint8_t *) wgt;

  MaxDeep_dramRead_actions_t read_actions;
  read_actions.param_size_bytes = burst_aligned_out_size * sizeof(uint32_t);
  read_actions.param_start_bytes = (burst_aligned_inp_size + burst_aligned_wgt_size) * sizeof(uint32_t);
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

  bool pass = true;
  for (int i = 0; i < out_size; i ++) {
    if (out[i] != expected_out[i]) {
      fprintf(stderr,
          "out[%3d] = %u is different from expected result %u\n",
          i, out[i], expected_out[i]);
      pass = false;
    }
  }
  if (pass)
    printf("TEST PASSED\n");
  else
    printf("TEST FAILED\n");

  free(inp);
  free(wgt);
  free(out);

  return pass;
}
}

}

#endif
