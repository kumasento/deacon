#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <ctime>

#include "Maxfiles.h"

int main(int argc, char *argv[]) {

  srand(42);

  max_file_t* max_file = ConvTwoLayers_init();
  max_engine_t* engine = max_load(max_file, "*");

  // load constants
  uint64_t conv0_H = max_get_constant_uint64t(max_file, "conv0_H");
  uint64_t conv0_W = max_get_constant_uint64t(max_file, "conv0_W");
  uint64_t conv0_C = max_get_constant_uint64t(max_file, "conv0_C");
  uint64_t conv0_F = max_get_constant_uint64t(max_file, "conv0_F");
  uint64_t conv0_K = max_get_constant_uint64t(max_file, "conv0_K");

  uint64_t conv1_H = max_get_constant_uint64t(max_file, "conv1_H");
  uint64_t conv1_W = max_get_constant_uint64t(max_file, "conv1_W");
  uint64_t conv1_C = max_get_constant_uint64t(max_file, "conv1_C");
  uint64_t conv1_F = max_get_constant_uint64t(max_file, "conv1_F");
  uint64_t conv1_K = max_get_constant_uint64t(max_file, "conv1_K");

  uint64_t num_batches = 10;

  uint64_t ifmap_num_elems = conv0_H * conv0_W * conv0_C * num_batches;
  uint64_t coeff_0_num_elems = conv0_F * conv0_C * conv0_K * conv0_K * num_batches;
  uint64_t coeff_1_num_elems = conv1_F * conv1_C * conv1_K * conv1_K * num_batches;
  uint64_t ofmap_num_elems = (conv1_H - conv1_K + 1) * (conv1_W - conv1_K + 1) * conv1_F * num_batches;

  int32_t *ifmap = (int32_t *) malloc(sizeof(int32_t) * ifmap_num_elems);
  int32_t *coeff_0 = (int32_t *) malloc(sizeof(int32_t) * coeff_0_num_elems);
  int32_t *coeff_1 = (int32_t *) malloc(sizeof(int32_t) * coeff_1_num_elems);
  int32_t *ofmap = (int32_t *) malloc(sizeof(int32_t) * ofmap_num_elems);

  for (uint64_t i = 0; i < ifmap_num_elems; i ++)
    ifmap[i] = (rand() % 10) - 5;
  for (uint64_t i = 0; i < coeff_0_num_elems; i ++)
    coeff_0[i] = (rand() % 10) - 5;
  for (uint64_t i = 0; i < coeff_1_num_elems; i ++)
    coeff_1[i] = (rand() % 10) - 5;

  ConvTwoLayers_actions_t actions;
  actions.param_num_batches = num_batches;
  actions.instream_coeff_0 = (const int32_t *) coeff_0;
  actions.instream_coeff_1 = (const int32_t *) coeff_1;
  actions.instream_ifmap = (const int32_t *) ifmap;
  actions.outstream_ofmap = ofmap;

  printf("Running ...\n");
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < 1; i ++)
    ConvTwoLayers_run(engine, &actions);
  end = std::chrono::system_clock::now();
  printf("Done\n");

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() / 1 << "s\n";
  uint64_t num_ops = conv1_H * conv1_W * conv0_C * conv0_F * conv0_K * conv0_K * 2;
  num_ops += (conv1_H - conv1_K + 1) * (conv1_W - conv1_K + 1) * conv1_C * conv1_F * conv1_K * conv1_K * 2;
  std::cout << "GOP/s: " << num_ops * num_batches * 1e-9 / elapsed_seconds.count() * 1 << std::endl;

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
