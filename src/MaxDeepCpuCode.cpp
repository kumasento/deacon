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

  int num_iters = 100;
  uint64_t num_elems = pow(1024, 3);

  uint32_t *data_inp = (uint32_t *) malloc(sizeof(uint32_t) * num_elems);
  uint32_t *data_out = (uint32_t *) malloc(sizeof(uint32_t) * num_elems);

  for (uint64_t i = 0; i < num_elems; i ++)
    data_inp[i] = (uint32_t) 1;

  max_file_t *maxfile = MaxDeep_init();
  max_engine_t *engine = max_load(maxfile, "local:*");

  MaxDeep_actions_t actions;
  actions.param_num_elems = num_elems;
  actions.instream_cpu_inp = data_inp;
  actions.outstream_cpu_out = data_out;

  std::cout << "Run start" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iters; i ++)
    MaxDeep_run(engine, &actions);

  double clock_diff = dfesnippets::timing::clock_diff(start);

  std::cout << "Total time:      " << clock_diff << " seconds." << std::endl;
  std::cout << "Per Iter time:   " << clock_diff / num_iters << " seconds." << std::endl;
  std::cout << "Per Item time:   " << clock_diff / num_iters / num_elems << " seconds." << std::endl;
  std::cout << "Per Item freq:   " << 1 / (clock_diff / num_iters / num_elems) * 1e-6 << " MHz" << std::endl;
  std::cout << "Per Byte time:   " << clock_diff / num_iters / num_elems / 4 << " seconds." << std::endl;
  std::cout << "Esti. bandwidth: " << 1 / (clock_diff / num_iters / num_elems / 4) * 1e-9 << " GB/s." << std::endl;

  for (uint32_t i = 0; i < num_elems; i ++)
    if (data_out[i] != 1)
      std::cout << "ERROR! Current output data is " << data_out[i] << std::endl;

  max_unload(engine);

  return 0;
}
