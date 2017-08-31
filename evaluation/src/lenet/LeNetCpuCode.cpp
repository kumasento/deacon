#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <getopt.h>

#include "Maxfiles.h"

int main(int argc, char *argv[]) {
  char c;

  uint64_t batch_size = 1;
  uint64_t num_iters = 1;
  while ((c = getopt(argc, argv, "i:n:")) != -1)
    switch (c) {
      case 'i':
        num_iters = atoi(optarg);
        break;
      case 'n':
        batch_size = atoi(optarg);
        break;
      default:
        exit(1);
    }

  max_file_t *max_file = LeNet_init();
  max_engine_t* engine = max_load(max_file, "*");

  LeNet_actions_t actions;
  actions.param_batch_size = batch_size;

  printf("Running ...\n");
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < (int) num_iters; i ++)
    LeNet_run(engine, &actions);
  end = std::chrono::system_clock::now();
  printf("Done\n");

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time:            " << elapsed_seconds.count() / num_iters << "s\n";
  std::cout << "elapsed time (per item): " << elapsed_seconds.count() / num_iters / batch_size << "s\n";

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
