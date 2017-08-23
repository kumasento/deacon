#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>

#include "Maxfiles.h"

int main(int argc, char *argv[]) {

  max_file_t *max_file = DepthwiseSeparableConvLayer_init();
  max_engine_t* engine = max_load(max_file, "*");

  uint64_t H = max_get_constant_uint64t(max_file, "conv_H");
  uint64_t W = max_get_constant_uint64t(max_file, "conv_W");
  uint64_t C = max_get_constant_uint64t(max_file, "conv_C");
  uint64_t F = max_get_constant_uint64t(max_file, "conv_F");
  uint64_t K = max_get_constant_uint64t(max_file, "conv_K");
  
  uint64_t batch_size = 1;
  uint64_t num_iters = 1;

  uint64_t ifmap_num_elems = H * W * C * batch_size;
  uint64_t depthwise_coeff_num_elems = C * K * K * batch_size;
  uint64_t pointwise_coeff_num_elems = C * F * batch_size;
  uint64_t ofmap_num_elems = (H - K + 1) * (W - K + 1) * F * batch_size;

  int32_t *ifmap = (int32_t *) malloc(sizeof(int32_t) * ifmap_num_elems);
  int32_t *depthwise_coeff = (int32_t *) malloc(sizeof(int32_t) * depthwise_coeff_num_elems);
  int32_t *pointwise_coeff = (int32_t *) malloc(sizeof(int32_t) * pointwise_coeff_num_elems);
  int32_t *ofmap = (int32_t *) malloc(sizeof(int32_t) * ofmap_num_elems);

  // for (uint64_t i = 0; i < ifmap_num_elems; i ++)
  //   ifmap[i] = (rand() % 10) - 5;
  // for (uint64_t i = 0; i < coeff_0_num_elems; i ++)
  //   coeff_0[i] = (rand() % 10) - 5;

  DepthwiseSeparableConvLayer_actions_t actions;
  actions.param_batch_size = batch_size;
  // actions.instream_ifmap = (const int32_t *) ifmap;
  // actions.instream_coeff_0 = (const int32_t *) coeff_0;
  // actions.outstream_ofmap = ofmap;

  printf("Running ...\n");
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < num_iters; i ++)
    DepthwiseSeparableConvLayer_run(engine, &actions);
  end = std::chrono::system_clock::now();
  printf("Done\n");

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() / 1 << "s\n";
  uint64_t num_ops = H * W * C * (K * K + F) * 2;

  std::cout << "GOP/s: " << num_ops * batch_size * 1e-9 / elapsed_seconds.count() * num_iters << std::endl;

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}