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
#if defined(DESIGN_LOOPBACK) || defined(DESIGN_LOOPBACK_PADDED)
  actions.param_num_elems = inp_size;
#endif

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

  int64_t inp_size = 1000;
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
#if defined(DESIGN_MULT_ARRAY)
  actions.param_num_elems = inp_size;
#endif

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

/**
 * Run ONE_DIM_CONV test
 *
 * @author Ruizhe Zhao
 * @since 20/05/2017
 */
bool run_one_dim_conv_test(bool is_sim, max_file_t *maxfile, max_engine_t *engine) {
  printf("\x1B[32mTESTING\x1B[0m ONE_DIM_CONV ...\n");

  const int num_iters = (is_sim) ? 1 : 100;
  size_t burst_size = max_get_burst_size(maxfile, NULL);
  int num_pipes = (int) max_get_constant_uint64t(maxfile, "NUM_PIPES");
  int window_width = (int) max_get_constant_uint64t(maxfile, "ONE_DIM_CONV_WINDOW_WIDTH");
  
  int64_t num_of_lines = num_pipes * 2;
  int64_t line_width = 10;

  // Increase the total size if not in the simulation environment
  if (!is_sim) {
    if (num_of_lines < 100)
      num_of_lines *= 10;
    num_of_lines = num_of_lines * 1024 * 10;
    line_width = line_width * 10;
  }

  int64_t inp_size = num_of_lines * line_width;
  int64_t wgt_size = num_of_lines * window_width;
  int64_t out_size = num_of_lines * (line_width - window_width + 1);

  int64_t burst_aligned_inp_size = utils::burst_aligned_size(inp_size, sizeof(uint32_t), burst_size);
  int64_t burst_aligned_wgt_size = utils::burst_aligned_size(wgt_size, sizeof(uint32_t), burst_size);
  int64_t burst_aligned_out_size = utils::burst_aligned_size(out_size, sizeof(uint32_t), burst_size);

  printf("Allocating memory ...\n");
  uint32_t *inp = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_inp_size);
  uint32_t *inp_orig = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_inp_size);
  uint32_t *wgt = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_wgt_size);
  uint32_t *out = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_out_size);

  uint32_t *expected_out = (uint32_t *) malloc(sizeof(uint32_t) * out_size);
  if (!inp || !wgt || !out) {
    fprintf(stderr, "Cannot allocat streams\n");
    exit(-1);
  }

  printf("Initializing data ...\n");
  printf("Initializing INP ...\n");
  int inp_idx = 0;
  for (int i = 0; i < num_of_lines; i += num_pipes) {
    for (int j = 0; j < line_width; j ++) {
      for (int k = 0; k < num_pipes; k ++) {
        int line_idx = i + k;
        int val = line_idx * line_width + j + 1;
        inp[inp_idx ++] = val;
        inp_orig[line_idx * line_width + j] = val;
      }
    }
  }

  printf("Initializing WGT ...\n");
  for (int i = 0; i < (int) wgt_size; i++)
    wgt[i] = (uint32_t) i + 1;

  // expected output
  printf("Initializing OUT ...\n");
  int out_idx = 0;
  for (int i = 0; i < (int) num_of_lines; i += num_pipes) {
    for (int j = 0; j < (int) line_width; j ++) {
      if (j + window_width > line_width)
        break;

      for (int k = 0; k < num_pipes; k ++) {
        int line_idx = i + k;
        uint32_t sum = 0;
        for (int w = 0; w < window_width; w ++)
          sum += inp_orig[line_idx * line_width + j + w] * wgt[line_idx * window_width + w];
        expected_out[out_idx ++] = sum;
      }

    }
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
#if defined(DESIGN_ONE_DIM_CONV)
  actions.param_total_num_of_lines = num_of_lines;
  actions.param_line_width = line_width;
#endif

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

  double gop = out_size * window_width * 2 * 1e-9;
  double gops = gop / elapsed;
  printf("RUN TIME:  %lf s\n", elapsed);
  printf("FREQUENCY: %.2f MHz\n", inp_size / elapsed * 1e-6);
  printf("GOP/s:     %.2f\n", gops); 

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

class Conv2DTest {
public:
  Conv2DTest(int num_of_windows, bool sim, max_file_t *max_file, max_engine_t *max_engine) {
    this->num_of_windows = num_of_windows;
    this->sim = sim;
    this->max_file = max_file;
    this->max_engine = max_engine;

    size_t burst_size = max_get_burst_size(max_file, NULL);

    num_of_pipes  = (int) max_get_constant_uint64t(max_file, "NUM_PIPES");
    kernel_height = (int) max_get_constant_uint64t(max_file, "KERNEL_SIZE");
    kernel_width  = (int) max_get_constant_uint64t(max_file, "KERNEL_SIZE");
    
    inp_size = num_of_windows * kernel_height * kernel_width;
    wgt_size = num_of_windows * num_of_pipes * kernel_height * kernel_width;
    out_size = num_of_windows * num_of_pipes;

    burst_aligned_inp_size = utils::burst_aligned_size(inp_size, sizeof(uint32_t), burst_size);
    burst_aligned_wgt_size = utils::burst_aligned_size(wgt_size, sizeof(uint32_t), burst_size);
    burst_aligned_out_size = utils::burst_aligned_size(out_size, sizeof(uint32_t), burst_size);
  }

  /**
   * Main function for the test.
   *
   * @param num_iters number of iterations for hardware build test
   * @return boolean indicates whether the test passes or not
   */
  bool run(int num_hardware_iters) {
    printf("\x1B[32mTESTING\x1B[0m CONV2D ...\n");

    const int num_iters = (sim) ? 1 : num_hardware_iters;

    alloc_data();
    
    init_data();

    MaxDeep_dramWrite_actions_t inp_write_actions =
      create_write_actions(burst_aligned_inp_size, 0, inp);
    MaxDeep_dramWrite_actions_t wgt_write_actions =
      create_write_actions(burst_aligned_wgt_size, burst_aligned_inp_size, wgt);
    MaxDeep_dramRead_actions_t read_actions =
      create_read_actions(burst_aligned_out_size,
          burst_aligned_inp_size + burst_aligned_wgt_size,
          out);

    MaxDeep_actions_t actions;
#if defined(DESIGN_CONV2D)
    printf("num_of_windows: %d\n", num_of_windows);
    actions.param_num_of_windows = num_of_windows;
#endif

    printf("Writing to DRAM ...\n\n");
    MaxDeep_dramWrite_run(max_engine, &inp_write_actions);
    MaxDeep_dramWrite_run(max_engine, &wgt_write_actions);

    printf("Computing ...\n\n");

    struct timeval t0, t1;

    gettimeofday(&t0, NULL);
    for (int i = 0; i < num_iters; i ++)
      MaxDeep_run(max_engine, &actions);
    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    elapsed /= num_iters;

    double gop = wgt_size * 2 * 1e-9;
    double gops = gop / elapsed;
    printf("RUN TIME:  %lf s\n", elapsed);
    printf("FREQUENCY: %.2f MHz\n", inp_size / (kernel_height * kernel_width) / elapsed * 1e-6);
    printf("GOP/s:     %.2f\n", gops); 

    printf("Reading back ...\n\n");
    MaxDeep_dramRead_run(max_engine, &read_actions);
    for (int i = 0; i < 10; i ++)
      printf("out[%3d] = %u\n", i, out[i]);

    bool pass = check();

    free_data();

    return pass;
  }

private:
  bool sim;
  max_file_t *max_file;
  max_engine_t *max_engine;

  int num_of_windows;
  int num_of_pipes;
  int kernel_height;
  int kernel_width;
  int64_t inp_size, wgt_size, out_size;
  int64_t burst_aligned_inp_size, burst_aligned_wgt_size, burst_aligned_out_size;
  uint32_t *inp, *wgt, *out, *expected;

  void alloc_data() {
    printf("Allocating memory ...\n");
    inp = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_inp_size);
    wgt = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_wgt_size);
    out = (uint32_t *) malloc(sizeof(uint32_t) * burst_aligned_out_size);
    expected = (uint32_t *) malloc(sizeof(uint32_t) * out_size);
    if (!inp || !wgt || !out || !expected) {
      fprintf(stderr, "Cannot allocat streams\n");
      exit(-1);
    }
  }

  void free_data() {
    free(inp);
    free(wgt);
    free(out);
    free(expected);
  }

  void init_data() {
    printf("Initializing data ...\n");
    printf("Initializing INP ...\n");
    for (int i = 0; i < (int) inp_size; i ++)
      inp[i] = (uint32_t) i + 1;

    printf("Initializing WGT ...\n");
    for (int i = 0; i < (int) wgt_size; i++)
      wgt[i] = (uint32_t) i + 1;

    // expected output
    printf("Initializing OUT ...\n");
    for (int i = 0; i < (int) num_of_windows; i ++) {
      for (int j = 0; j < num_of_pipes; j ++) {
        uint32_t sum = 0;
        for (int k = 0; k < (int) kernel_height * kernel_width; k ++) {
          int inp_idx = i * kernel_height * kernel_width + k;
          int wgt_idx = i * num_of_pipes * kernel_height * kernel_width
            + j * kernel_height * kernel_width + k;
          sum += inp[inp_idx] * wgt[wgt_idx];
        }
        int out_idx = i * num_of_pipes + j;
        expected[out_idx] = sum;
      }
    }
  }

  bool check() {
    bool pass = true;
    for (int i = 0; i < out_size; i ++) {
      if (out[i] != expected[i]) {
        fprintf(stderr,
            "out[%3d] = %u is different from expected result %u\n",
            i, out[i], expected[i]);
        pass = false;
      }
    }
    if (pass)
      printf("TEST PASSED\n");
    else
      printf("TEST FAILED\n");
    return pass;
  }

  MaxDeep_dramWrite_actions_t create_write_actions(int64_t num_of_elems, int64_t start, uint32_t *ptr) {
    MaxDeep_dramWrite_actions_t write_actions;
    write_actions.param_size_bytes = num_of_elems * sizeof(uint32_t);
    write_actions.param_start_bytes = start * sizeof(uint32_t);
    write_actions.instream_fromcpu = (const uint8_t *) ptr;
    return write_actions;
  }

  MaxDeep_dramRead_actions_t create_read_actions(int64_t num_of_elems, int64_t start, uint32_t *ptr) {
    MaxDeep_dramRead_actions_t read_actions;
    read_actions.param_size_bytes = num_of_elems * sizeof(uint32_t);
    read_actions.param_start_bytes = start * sizeof(uint32_t);
    read_actions.outstream_tocpu = (uint8_t *) ptr;
    return read_actions;
  }
};

}

}

#endif
