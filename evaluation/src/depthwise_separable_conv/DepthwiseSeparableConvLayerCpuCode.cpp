/**
 * Evaluation of Depthwise Separable Convolution.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <getopt.h>

#include "Maxfiles.h"
#include "maxdeep/layers.h"
#include "maxdeep/utils.h"

#if BIT_WIDTH == 8
typedef int8_t data_t;
#elif BIT_WIDTH == 16
typedef int16_t data_t;
#elif BIT_WIDTH == 32
typedef int32_t data_t;
#endif

int main(int argc, char *argv[]) {
  srand(42);

  char c;

  uint64_t batch_size = 1;
  uint64_t num_iters = 1;
  while ((c = getopt(argc, argv, "i:n:")) != -1) {
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
  }

  max_file_t *max_file = DepthwiseSeparableConvLayer_init();
  max_engine_t *engine = max_load(max_file, "*");

  uint64_t H = max_get_constant_uint64t(max_file, "conv_H");
  uint64_t W = max_get_constant_uint64t(max_file, "conv_W");
  uint64_t C = max_get_constant_uint64t(max_file, "conv_C");
  uint64_t F = max_get_constant_uint64t(max_file, "conv_F");
  uint64_t K = max_get_constant_uint64t(max_file, "conv_K");

  printf("H = %lu\n", H);
  printf("W = %lu\n", W);
  printf("C = %lu\n", C);
  printf("F = %lu\n", F);
  printf("K = %lu\n", K);
  printf("batch_size = %lu\n", batch_size);

  // for (uint64_t i = 0; i < ifmap_num_elems; i ++)
  //   ifmap[i] = (rand() % 10) - 5;
  // for (uint64_t i = 0; i < coeff_0_num_elems; i ++)
  //   coeff_0[i] = (rand() % 10) - 5;

  printf("Initializing arrays ...\n");

  DepthwiseSeparableConvLayer_actions_t actions;
  actions.param_batch_size = batch_size;
#ifndef USE_DRAM
  uint64_t ifmap_num_elems = H * W * C * batch_size;
  uint64_t ofmap_num_elems = (H - K + 1) * (W - K + 1) * F * batch_size;

  auto ifmap = random_initialize<data_t>(ifmap_num_elems, 100);
  auto ofmap = create_array<data_t>(ofmap_num_elems);
  auto ofmap_golden = create_array<data_t>(ofmap_num_elems);

  actions.instream_ifmap = (const data_t *)ifmap;
  actions.outstream_ofmap = ofmap;

#ifndef DEPTHWISE_SEPARABLE_V2
  uint64_t depthwise_coeff_num_elems = C * K * K * batch_size;
  uint64_t pointwise_coeff_num_elems = C * F * batch_size;
  data_t *depthwise_coeff_0 =
      (data_t *)malloc(sizeof(data_t) * depthwise_coeff_num_elems);
  data_t *pointwise_coeff_0 =
      (data_t *)malloc(sizeof(data_t) * pointwise_coeff_num_elems);
  actions.instream_depthwise_coeff_0 = (const data_t *)depthwise_coeff_0;
  actions.instream_pointwise_coeff_0 = (const data_t *)pointwise_coeff_0;
#else
  uint64_t coeff_num_elems = C * K * K * (1 + F) * batch_size;
  auto coeff_0 = random_initialize<data_t>(coeff_num_elems, 100);
  actions.instream_coeff_0 = (const data_t *)coeff_0;

  dump_array("coeff.txt", coeff_0, coeff_num_elems);
#endif

#else
#error "Using DRAM is not supported yet"
#endif

  printf("Running golden function ...\n");
  depthwise_separable_conv_layer(ifmap, coeff_0, ofmap_golden, H, W, C, F, K,
                                 batch_size);

  printf("Running ...\n");
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < (int)num_iters; i++)
    DepthwiseSeparableConvLayer_run(engine, &actions);
  end = std::chrono::system_clock::now();
  printf("Done\n");

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() / 1 << "s\n";
  uint64_t num_ops = (((H - K + 1) * (W - K + 1) * C * K * K) +
                      ((H - K + 1) * (W - K + 1) * C * F)) *
                     2;
  uint64_t num_conv_ops = (H - K + 1) * (W - K + 1) * C * F * K * K * 2;

  std::cout << "GOP/s: "
            << num_ops *batch_size * 1e-9 / elapsed_seconds.count() * num_iters
            << std::endl;
  std::cout << "GOP/s: " << num_conv_ops *batch_size * 1e-9 /
                                elapsed_seconds.count() *
                                num_iters << " (CONV)" << std::endl;

  for (int i = 0; i < 10; i++) printf("ofmap[%5d] = %d\n", i, ofmap[i]);
  printf("Golden:\n");
  for (int i = 0; i < 10; i++) printf("ofmap[%5d] = %d\n", i, ofmap_golden[i]);

  printf("Running test ...\n");
  for (int i = 0; i < ofmap_num_elems; i++)
    if (ofmap[i] != ofmap_golden[i]) {
      fprintf(stderr, "ofmap doesn't matched at %d: %d != %d\n", i, ofmap[i],
              ofmap_golden[i]);
      exit(1);
    }

  printf("Test PASSED!\n");

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
