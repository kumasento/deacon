#include <getopt.h>
#include <glog/logging.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "Maxfiles.h"
#include "maxdeep/layers.h"

max_file_t *max_file;
max_engine_t *engine;

struct ConvSingleLayerRun {
  void run(max_engine_t *engine, ConvSingleLayer_actions_t *actions) {
    ConvSingleLayer_run(engine, actions);
  }
};

template <typename T>
void TestTiles(int N_TOH, int N_TOW, int N_TC, int N_TF) {
  // get tile size
  int P = 1, S = 1;

  auto DFE_TH = max_get_constant_uint64t(max_file, "conv_H");
  auto DFE_TW = max_get_constant_uint64t(max_file, "conv_W");
  auto TC = max_get_constant_uint64t(max_file, "conv_C");
  auto TF = max_get_constant_uint64t(max_file, "conv_F");
  auto K = static_cast<int>(max_get_constant_uint64t(max_file, "conv_K"));

  auto TH = GetConvLayerOutputDim(DFE_TH, K, 0, S);
  auto TW = GetConvLayerOutputDim(DFE_TW, K, 0, S);
  auto H = N_TOH * static_cast<int>(TH);
  auto W = N_TOW * static_cast<int>(TW);

  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);
  auto C = N_TC * static_cast<int>(TC);
  auto F = N_TF * static_cast<int>(TF);

  LOG(INFO) << C << " x " << H << " x " << W << " -> " << F << " x " << OH
            << " x " << OW;

  // initialise test array
  int max_val = 10, min_val = -10;

  auto input = CreateRandomArray<T>(C * H * W, min_val, max_val);
  auto weights = CreateRandomArray<T>(F * C * K * K, min_val, max_val);
  auto bias = CreateRandomArray<T>(F, min_val, max_val);
  auto output_cpu = std::vector<T>(F * OH * OW);
  auto output_dfe = std::vector<T>(F * OH * OW);

  ConvLayerCpu(input, weights, bias, output_cpu, H, W, C, F, K, P, S, false);
  ConvLayerDfe<T, ConvSingleLayer_actions_t, ConvSingleLayerRun>(
      input, weights, bias, output_dfe, H, W, C, F, K, P, S, max_file, engine);

  for (int i = 0; i < (int)(TF * TH * TW); i++)
    if (output_cpu[i] != output_dfe[i]) {
      fprintf(stderr, "Result mis-matched at %6d: cpu %6d dfe %6d\n", i,
              output_cpu[i], output_dfe[i]);
      exit(1);
    }

  LOG(INFO) << "TEST single tile: " << ANSI_COLOR_GREEN << "PASSED!"
            << ANSI_COLOR_RESET;
}

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  srand(42);

  max_file = ConvSingleLayer_init();
  engine = max_load(max_file, "*");

  // run test on single tile
  TestTiles<int16_t>(1, 1, 1, 1);
  TestTiles<int16_t>(2, 2, 1, 1);
  TestTiles<int16_t>(1, 1, 2, 2);
  TestTiles<int16_t>(2, 2, 2, 2);

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
