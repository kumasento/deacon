#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>

#include "Maxfiles.h"

template <typename T>
void PointwiseConvolutionCpu(std::vector<T> &ifmap, std::vector<T> &weights,
                             std::vector<T> &bias, std::vector<T> &ofmap,
                             int height, int width, int in_depth,
                             int out_depth) {
  for (int f = 0; f < out_depth; f++) {
    auto bias_val = bias[f];
    for (int c = 0; c < in_depth; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          auto ofmap_idx = f * height * width + h * width + w;
          auto ifmap_val = ifmap[c * height * width + h * width + w];
          auto weights_val = weights[f * in_depth + c];

          if (c == 0) ofmap[ofmap_idx] = bias_val;

          ofmap[ofmap_idx] += ifmap_val * weights_val;
        }
      }
    }
  }
}

template <typename T>
void PointwiseConvolutionDfe(std::vector<T> &ifmap, std::vector<T> &weights,
                             std::vector<T> &bias, std::vector<T> &ofmap,
                             int height, int width, int in_depth, int out_depth,
                             int tile_height, int tile_width, int tile_in_depth,
                             int tile_out_depth, int par_width,
                             int par_in_depth, int par_out_depth) {}

typedef float T;

int main(int argc, char *argv[]) {
  const int height = 128;
  const int width = 128;
  const int in_depth = 512;
  const int out_depth = 512;

  std::vector<T> ifmap(in_depth * height * width);
  std::vector<T> weights(out_depth * in_depth);
  std::vector<T> bias(out_depth);
  std::vector<T> ofmap(out_depth * height * width);

  for (int i = 0; i < (int)ifmap.size(); i++)
    ifmap[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)weights.size(); i++)
    weights[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)bias.size(); i++)
    bias[i] = (T)((double)rand() / RAND_MAX);

  PointwiseConvolutionCpu<T>(ifmap, weights, bias, ofmap, height, width,
                             in_depth, out_depth);

  for (int i = 0; i < (int)ofmap.size(); i++)
    std::cout << ofmap[i] << std::endl;

  max_file_t *max_file = PointwiseConvolution_init();
  max_engine_t *engine = max_load(max_file, "*");

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
