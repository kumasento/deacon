/**
 * This header file contains several test runners for the MaxDeep project.
 */

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
  printf("\x1B[32mTESTING\x1B[0m loopback ...\n");

  const int num_iters = 1;
  size_t burst_size = max_get_burst_size(maxfile, NULL);

  int64_t inp_size = burst_size;
  // Increase the total size if not in the simulation environment
  if (!is_sim)
    inp_size = inp_size * 1024 * 1024;

  int64_t wgt_size = inp_size;
  int64_t out_size = inp_size;

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
  free(wgt);
  free(out);
}

}

}
