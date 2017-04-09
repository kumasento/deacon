#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <getopt.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <chrono>

#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

#include "dfesnippets/Timing.hpp"

int main(int argc, char *argv[]) {

  int num_iters = 256;
  int num_elems = 1024 * 1024;
  uint32_t *data_inp = (uint32_t *) malloc(sizeof(uint32_t) * num_elems);
  uint32_t *data_out = (uint32_t *) malloc(sizeof(uint32_t) * num_elems);

  for (int i = 0; i < num_elems; i ++)
    data_inp[i] = (uint32_t) i;

  max_file_t *maxfile = MaxDeep_init();
  max_engine_t *engine = max_load(maxfile, "local:*");

  MaxDeep_actions_t actions;
  actions.param_num_elems = num_elems;
  actions.instream_cpu_inp = data_inp;
  actions.outstream_cpu_out = data_out;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iters; i ++)
    MaxDeep_run(engine, &actions);
  double clock_diff = dfesnippets::timing::clock_diff(start);
  std::cout << "Total time:       " << clock_diff << " seconds." << std::endl;
  std::cout << "Total trans time: " << num_iters * num_elems * sizeof(uint32_t) / ((double) 2 * 1024 * 1024 * 1024) << " seconds." << std::endl;
  std::cout << "Per Iter time:    " << clock_diff / num_iters << " seconds." << std::endl;
  std::cout << "Per Item time:    " << clock_diff / num_iters / num_elems << " seconds." << std::endl;

  max_unload(engine);

  return 0;
}
