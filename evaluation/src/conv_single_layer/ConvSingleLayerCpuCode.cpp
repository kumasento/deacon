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

struct Dfe {
  typedef ConvSingleLayer_actions_t dfe_run_actions_t;
  typedef ConvSingleLayer_dramRead_actions_t dram_read_actions_t;
  typedef ConvSingleLayer_dramWrite_actions_t dram_write_actions_t;

  static void Run(max_engine_t *engine, dfe_run_actions_t *actions) {
    ConvSingleLayer_run(engine, actions);
  }
  static void ReadDRAM(max_engine_t *engine, dram_read_actions_t *actions) {
    ConvSingleLayer_dramRead_run(engine, actions);
  }
  static void WriteDRAM(max_engine_t *engine, dram_write_actions_t *actions) {
    ConvSingleLayer_dramWrite_run(engine, actions);
  }
};

template <typename T>
void TestTiles(int N_TOH, int N_TOW, int N_TC, int N_TF, int D_H = 0,
               int D_W = 0, int D_C = 0, int D_F = 0) {
  // get tile size
  int P = 1, S = 1;

  auto DFE_TH = max_get_constant_uint64t(max_file, "conv_H");
  auto DFE_TW = max_get_constant_uint64t(max_file, "conv_W");
  auto TC = max_get_constant_uint64t(max_file, "conv_C");
  auto TF = max_get_constant_uint64t(max_file, "conv_F");
  auto K = static_cast<int>(max_get_constant_uint64t(max_file, "conv_K"));
  auto NUM_FRAC_BITS = max_get_constant_uint64t(max_file, "conv_num_frac_bits");

  auto TH = GetConvLayerOutputDim(DFE_TH, K, 0, S);
  auto TW = GetConvLayerOutputDim(DFE_TW, K, 0, S);
  auto H = N_TOH * static_cast<int>(TH) + D_H;
  auto W = N_TOW * static_cast<int>(TW) + D_W;

  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);
  auto C = N_TC * static_cast<int>(TC) + D_C;
  auto F = N_TF * static_cast<int>(TF) + D_F;

  LOG(INFO) << C << " x " << H << " x " << W << " -> " << F << " x " << OH
            << " x " << OW;

  // initialise test array
  // use max_val and min_val to prevent overflow
  float max_val = 2, min_val = -2;

  auto input = CreateRandomArray<float>(C * H * W, min_val, max_val);
  auto weights = CreateRandomArray<float>(F * C * K * K, min_val, max_val);
  auto bias = CreateRandomArray<float>(F, min_val, max_val);

  auto input_dfe = FloatToFixed<T>(input, NUM_FRAC_BITS);
  auto weights_dfe = FloatToFixed<T>(weights, NUM_FRAC_BITS);
  auto bias_dfe = FloatToFixed<T>(bias, NUM_FRAC_BITS);

  auto output_cpu = std::vector<T>(F * OH * OW);
  auto output_dfe = std::vector<T>(F * OH * OW);

  LOG(INFO) << "Running CPU reference ...";
  ConvLayerCpu<T>(input_dfe, weights_dfe, bias_dfe, output_cpu, H, W, C, F, K,
                  P, S, false, NUM_FRAC_BITS);

  LOG(INFO) << "Running DFE reference ...";
  ConvLayerDfe<T, Dfe>(input_dfe, weights_dfe, bias_dfe, output_dfe, H, W, C, F,
                       K, P, S, max_file, engine);

  // skip tests for Winograd, due to the low-accuracy caused by 1/6
  bool failed = false;
  auto output_cpu_float = FixedToFloat(output_cpu, NUM_FRAC_BITS);
  auto output_dfe_float = FixedToFloat(output_dfe, NUM_FRAC_BITS);
  for (int i = 0; i < (int)(TF * TH * TW); i++) {
    auto diff = std::abs((output_cpu_float[i] - output_dfe_float[i]) /
                         output_cpu_float[i]);

    if (diff > 0.1f) {
      fprintf(stderr,
              "Result mis-matched at %6d: cpu %20.6f dfe %20.6f diff %10.6f\n",
              i, output_cpu_float[i], output_dfe_float[i], diff);
      failed = true;
      exit(1);
    }
  }

  if (!failed)
    LOG(INFO) << "TEST single tile: " << ANSI_COLOR_GREEN << "PASSED!"
              << ANSI_COLOR_RESET;
}

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  srand(42);

  max_file = ConvSingleLayer_init();
  engine = max_load(max_file, "*");

  auto DTYPE = max_get_constant_string(max_file, "conv_dtype");

#ifdef __SIM__
  if (!strcmp(DTYPE, "float")) {
    TestTiles<float>(1, 1, 1, 1);
    TestTiles<float>(2, 2, 2, 2);
  } else if (!strcmp(DTYPE, "fixed")) {
    TestTiles<int16_t>(1, 1, 1, 1);
    TestTiles<int16_t>(2, 2, 2, 2);
  }
#else
  TestTiles<int16_t>(1, 1, 2, 2);
#endif

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
