#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <vector>
#include <getopt.h>
#include <glog/logging.h>

#include "Maxfiles.h"
#include "maxdeep/layers.h"

typedef int16_t T;

max_file_t *max_file;
max_engine_t *engine;

/*! Tiled convolution layer on DFE.
 *
 * We assume that all inputs are already tiled,
 * and we output tiled results.
 * Therefore, for cases when the input shape equals to the tile shape,
 * we don't need any preprocessing and post-processing.
 *
 * NOTE that there is no explicit padding in the DFE -
 * we just make sure that the tiled_input has padded zero values.
 */
template <typename T>
void ConvLayerTiledDfe(std::vector<T> &tiled_input,
                       std::vector<T> &tiled_weights,
                       std::vector<T> &tiled_bias, std::vector<T> &tiled_output,
                       int H, int W, int C, int F, int K, int P, int S, int TH,
                       int TW, int TC, int TF) {
  // get number of total tiles
  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);
  auto NTH = GetNumTiles(OH, TH);
  auto NTW = GetNumTiles(OW, TW);
  auto NTC = GetNumTiles(C, TC);
  auto NTF = GetNumTiles(F, TF);
  auto NT = NTH * NTW * NTC * NTF;

  ConvSingleLayer_actions_t actions;
  actions.param_batch_size = NT;
  actions.instream_ifmap = tiled_input.data();
  actions.instream_coeff_0 = tiled_weights.data();
  actions.outstream_ofmap = reinterpret_cast<T *>(tiled_output.data());

  std::cout << "Running convolution layer on DFE ..." << std::endl;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  ConvSingleLayer_run(engine, &actions);
  end = std::chrono::system_clock::now();
  std::cout << "Done" << std::endl;

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
  auto gflops =
      (2 * OH * OW * C * F * K * K) / (elapsed_seconds.count()) * 1e-9;
  std::cout << "GFLOPS: " << gflops << std::endl;
}

template <typename T>
void TestSingleTile() {
  // get tile size
  const int P = 1, S = 1;
  auto DFE_TH = max_get_constant_uint64t(max_file, "conv_H");
  auto DFE_TW = max_get_constant_uint64t(max_file, "conv_W");
  auto TC = max_get_constant_uint64t(max_file, "conv_C");
  auto TF = max_get_constant_uint64t(max_file, "conv_F");
  auto K = max_get_constant_uint64t(max_file, "conv_K");
  auto TH = GetConvLayerOutputDim(DFE_TH, K, 0, S);
  auto TW = GetConvLayerOutputDim(DFE_TW, K, 0, S);
  auto H = TH;
  auto W = TW;
  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);
  auto C = TC;
  auto F = TF;

#if 0
  std::cout << "DFE Tile height: " << DFE_TH << std::endl;
  std::cout << "DFE Tile width: " << DFE_TW << std::endl;
  std::cout << "Tile input height: " << TH << std::endl;
  std::cout << "Tile input width: " << TW << std::endl;
  std::cout << "Tile input channels: " << TC << std::endl;
  std::cout << "Tile output height: " << TH << std::endl;
  std::cout << "Tile output width: " << TW << std::endl;
  std::cout << "Tile output channels: " << TF << std::endl;
#endif

  // initialise test array
  int max_val = 5, min_val = 0;

  auto input = CreateRandomArray<T>(C * H * W, min_val, max_val);
  auto weights = CreateRandomArray<T>(F * C * K * K, min_val, max_val);
  auto bias = CreateRandomArray<T>(F, min_val, max_val);
  auto output_cpu = std::vector<T>(F * OH * OW);
  auto output_dfe = std::vector<T>(F * OH * OW);

  auto tiled_input = CreateConvLayerTiledInput<T>(input, H, W, C, F, K, P, S,
                                                  TH, TW, TC, TF, true);
  CHECK_EQ(tiled_input.size(), DFE_TH * DFE_TW * C);

  ConvLayerCpu(input, weights, bias, output_cpu, H, W, C, F, K, P, S, false);
  ConvLayerTiledDfe(tiled_input, weights, bias, output_dfe, H, W, C, F, K, P, S,
                    TH, TW, TC, TF);

  for (int i = 0; i < (int)(TF * TH * TW); i++)
    if (output_cpu[i] != output_dfe[i]) {
      fprintf(stderr, "Result mis-matched at %6d: cpu %6d dfe %6d\n", i,
              output_cpu[i], output_dfe[i]);
      exit(1);
    }

  printf("TEST single tile: " ANSI_COLOR_GREEN "PASSED!\n" ANSI_COLOR_RESET);
}

int main(int argc, char *argv[]) {
  max_file = ConvSingleLayer_init();
  engine = max_load(max_file, "*");

  // run test on single tile
  TestSingleTile<T>();

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
