#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <glog/logging.h>

#include "Maxfiles.h"

int get_num_tiles(int full, int tile) { return (int)ceil((double)full / tile); }

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
std::vector<std::vector<T>> PrepareTiledIfmap(std::vector<T> &ifmap, int height,
                                              int width, int in_depth,
                                              int tile_height, int tile_width,
                                              int tile_in_depth,
                                              int par_width = 1,
                                              int par_in_depth = 1) {
  auto num_tiles_height = get_num_tiles(height, tile_height);
  auto num_tiles_width = get_num_tiles(width, tile_width);
  auto num_tiles_in_depth = get_num_tiles(in_depth, tile_in_depth);
  auto num_tiles = num_tiles_height * num_tiles_width * num_tiles_in_depth;
  auto tile_num_elems = tile_height * tile_width * tile_in_depth;

  std::vector<std::vector<T>> tiled(num_tiles);
  for (int i = 0; i < num_tiles; i++) tiled[i].resize(tile_num_elems);

  for (int tc = 0; tc < num_tiles_in_depth; tc++) {
    for (int th = 0; th < num_tiles_height; th++) {
      for (int tw = 0; tw < num_tiles_width; tw++) {
        auto ic = tc * tile_in_depth;
        auto ih = th * tile_height;
        auto iw = tw * tile_width;
        auto tile_idx = (tc * num_tiles_height * num_tiles_width +
                         th * num_tiles_width + tw);

        for (int c = 0; c < tile_in_depth; c += par_in_depth) {
          for (int h = 0; h < tile_height; h++) {
            for (int w = 0; w < tile_width; w += par_width) {
              for (int ci = 0; ci < par_in_depth; ci++) {
                for (int wi = 0; wi < par_width; wi++) {
                  auto pc = c / par_in_depth;
                  auto pw = w / par_width;
                  auto tiled_idx = ((pc * tile_height * tile_width / par_width +
                                     h * tile_width / par_width + pw) *
                                    (par_width * par_in_depth)) +
                                   (ci * par_width + wi);
                  auto ifmap_idx = (ic + c + ci) * height * width +
                                   (ih + h) * width + (iw + w + wi);

                  if (ic + c + ci >= in_depth || ih + h >= height ||
                      iw + w + wi >= width)
                    tiled[tile_idx][tiled_idx] = (T)0.0f;

                  tiled[tile_idx][tiled_idx] = ifmap[ifmap_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  return tiled;
}

template <typename T>
std::vector<std::vector<T>> PrepareTiledWeights(
    std::vector<T> &weights, int in_depth, int out_depth, int tile_in_depth,
    int tile_out_depth, int par_in_depth = 1, int par_out_depth = 1) {
  auto num_tiles_in_depth = get_num_tiles(in_depth, tile_in_depth);
  auto num_tiles_out_depth = get_num_tiles(out_depth, tile_out_depth);
  auto num_tiles = num_tiles_in_depth * num_tiles_out_depth;
  auto tile_num_elems = tile_in_depth * tile_out_depth;

  std::vector<std::vector<T>> tiled(num_tiles);
  for (int i = 0; i < num_tiles; i++) tiled[i].resize(tile_num_elems);

  for (int tf = 0; tf < num_tiles_out_depth; tf++) {
    for (int tc = 0; tc < num_tiles_in_depth; tc++) {
      auto sf = tf * tile_out_depth;
      auto sc = tc * tile_in_depth;
      auto tile_idx = (tf * num_tiles_in_depth + tc);

      for (int f = 0; f < tile_out_depth; f += par_out_depth) {
        for (int c = 0; c < tile_in_depth; c += par_in_depth) {
          for (int fi = 0; fi < par_out_depth; fi++) {
            for (int ci = 0; ci < par_in_depth; ci++) {
              auto pf = f / par_out_depth;
              auto pc = c / par_in_depth;
              auto tiled_idx = ((pf * tile_in_depth / par_in_depth + pc) *
                                (par_in_depth * par_out_depth)) +
                               (fi * par_in_depth + ci);
              auto rf = sf + f + fi;
              auto rc = sc + c + ci;
              auto weights_idx = rf * in_depth + rc;

              tiled[tile_idx][tiled_idx] = (rc >= in_depth || rf >= out_depth)
                                               ? (T)0.0f
                                               : weights[weights_idx];
            }
          }
        }
      }
    }
  }

  return tiled;
}

template <typename T>
std::vector<std::vector<T>> PrepareTiledBias(std::vector<T> &bias,
                                             int out_depth,
                                             int tile_out_depth) {
  auto num_tiles_out_depth = get_num_tiles(out_depth, tile_out_depth);
  auto num_tiles = num_tiles_out_depth;
  auto tile_num_elems = tile_out_depth;

  std::vector<std::vector<T>> tiled(num_tiles);
  for (int i = 0; i < num_tiles; i++) tiled[i].resize(tile_num_elems);

  for (int tf = 0; tf < num_tiles_out_depth; tf++) {
    auto sf = tf * tile_out_depth;
    auto tile_idx = tf;

    for (int f = 0; f < tile_out_depth; f++) {
      auto tiled_idx = f;
      auto bias_idx = sf + f;

      if (sf + f >= out_depth) tiled[tile_idx][tiled_idx] = (T)0.0f;

      tiled[tile_idx][tiled_idx] = bias[bias_idx];
    }
  }

  return tiled;
}

template <typename T>
void PointwiseConvolutionTiledCpu(std::vector<T> &ifmap,
                                  std::vector<T> &weights, std::vector<T> &bias,
                                  std::vector<T> &ofmap, int height, int width,
                                  int in_depth, int out_depth, int tile_height,
                                  int tile_width, int tile_in_depth,
                                  int tile_out_depth) {
  // split input into tiles
  auto num_tiles_height = get_num_tiles(height, tile_height);
  auto num_tiles_width = get_num_tiles(width, tile_width);
  auto num_tiles_in_depth = get_num_tiles(in_depth, tile_in_depth);
  auto num_tiles_out_depth = get_num_tiles(out_depth, tile_out_depth);

  printf("Number of tiles: %3d x %3d x %3d x %3d\n", num_tiles_out_depth,
         num_tiles_in_depth, num_tiles_height, num_tiles_width);

  // tiled data
  LOG(INFO) << "Preparing tiled ifmap ...";
  auto tiled_ifmap = PrepareTiledIfmap<T>(
      ifmap, height, width, in_depth, tile_height, tile_width, tile_in_depth);
  LOG(INFO) << "Preparing tiled weights ...";
  auto tiled_weights = PrepareTiledWeights<T>(weights, in_depth, out_depth,
                                              tile_in_depth, tile_out_depth);
  LOG(INFO) << "Preparing tiled bias ...";
  auto tiled_bias = PrepareTiledBias<T>(bias, out_depth, tile_out_depth);

  std::vector<T> ofmap_tile(tile_out_depth * tile_height * tile_width);
  std::vector<T> zero_bias_tile(tile_out_depth);

  LOG(INFO) << "Running tiled computation ...";
  for (int tf = 0; tf < num_tiles_out_depth; tf++) {
    for (int tc = 0; tc < num_tiles_in_depth; tc++) {
      for (int th = 0; th < num_tiles_height; th++) {
        for (int tw = 0; tw < num_tiles_width; tw++) {
          auto ifmap_tile_idx = tc * num_tiles_height * num_tiles_width +
                                th * num_tiles_width + tw;
          auto weights_tile_idx = tf * num_tiles_in_depth + tc;
          auto bias_tile_idx = tf;

          PointwiseConvolutionCpu(
              tiled_ifmap[ifmap_tile_idx], tiled_weights[weights_tile_idx],
              ((tc == 0) ? tiled_bias[bias_tile_idx] : zero_bias_tile),
              ofmap_tile, tile_height, tile_width, tile_in_depth,
              tile_out_depth);

          auto sf = tf * tile_out_depth;
          auto sh = th * tile_height;
          auto sw = tw * tile_width;

          for (int f = 0; f < tile_out_depth; f++) {
            for (int h = 0; h < tile_height; h++) {
              for (int w = 0; w < tile_width; w++) {
                if ((f + sf) >= out_depth || (h + sh) >= height ||
                    (w + sw) >= width)
                  continue;

                auto ofmap_idx =
                    (sf + f) * height * width + (sh + h) * width + (sw + w);
                ofmap[ofmap_idx] += ofmap_tile
                    [f * tile_height * tile_width + h * tile_width + w];
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void PointwiseConvolutionDfe(max_engine_t *engine, std::vector<T> &ifmap,
                             std::vector<T> &weights, std::vector<T> &bias,
                             std::vector<T> &ofmap, int height, int width,
                             int in_depth, int out_depth, int tile_height,
                             int tile_width, int tile_in_depth,
                             int tile_out_depth, int par_width,
                             int par_in_depth, int par_out_depth) {
  // split input into tiles
  auto num_tiles_height = get_num_tiles(height, tile_height);
  auto num_tiles_width = get_num_tiles(width, tile_width);
  auto num_tiles_in_depth = get_num_tiles(in_depth, tile_in_depth);
  auto num_tiles_out_depth = get_num_tiles(out_depth, tile_out_depth);

  printf("Number of tiles: %3d x %3d x %3d x %3d\n", num_tiles_out_depth,
         num_tiles_in_depth, num_tiles_height, num_tiles_width);

  // tiled data
  LOG(INFO) << "Preparing tiled ifmap ...";
  auto tiled_ifmap =
      PrepareTiledIfmap<T>(ifmap, height, width, in_depth, tile_height,
                           tile_width, tile_in_depth, par_width, par_in_depth);
  LOG(INFO) << "Preparing tiled weights ...";
  auto tiled_weights =
      PrepareTiledWeights<T>(weights, in_depth, out_depth, tile_in_depth,
                             tile_out_depth, par_in_depth, par_out_depth);
  LOG(INFO) << "Preparing tiled bias ...";
  auto tiled_bias = PrepareTiledBias<T>(bias, out_depth, tile_out_depth);

  std::vector<T> ofmap_tile(tile_out_depth * tile_height * tile_width);
  std::vector<T> zero_bias_tile(tile_out_depth);

  LOG(INFO) << "Running tiled computation ...";
  for (int tf = 0; tf < num_tiles_out_depth; tf++) {
    for (int tc = 0; tc < num_tiles_in_depth; tc++) {
      for (int th = 0; th < num_tiles_height; th++) {
        for (int tw = 0; tw < num_tiles_width; tw++) {
          auto ifmap_tile_idx = tc * num_tiles_height * num_tiles_width +
                                th * num_tiles_width + tw;
          auto weights_tile_idx = tf * num_tiles_in_depth + tc;
          auto bias_tile_idx = tf;

          PointwiseConvolution_actions_t actions;
          actions.param_N = 1;
          actions.instream_ifmap = tiled_ifmap[ifmap_tile_idx].data();
          actions.instream_weights = tiled_weights[weights_tile_idx].data();
          actions.instream_bias =
              ((tc == 0) ? tiled_bias[bias_tile_idx] : zero_bias_tile).data();
          actions.outstream_ofmap = (T *)ofmap_tile.data();

          PointwiseConvolution_run(engine, &actions);

          auto sf = tf * tile_out_depth;
          auto sh = th * tile_height;
          auto sw = tw * tile_width;

          for (int f = 0; f < tile_out_depth; f += par_out_depth) {
            for (int h = 0; h < tile_height; h++) {
              for (int w = 0; w < tile_width; w += par_width) {
                for (int fi = 0; fi < par_out_depth; fi++) {
                  for (int wi = 0; wi < par_width; wi++) {
                    if ((f + sf + fi) >= out_depth || (h + sh) >= height ||
                        (w + sw + wi) >= width)
                      continue;

                    auto pf = f / par_out_depth;
                    auto pw = w / par_width;
                    auto ofmap_idx = (sf + f + fi) * height * width +
                                     (sh + h) * width + (sw + w + wi);
                    auto tiled_idx =
                        ((pf * tile_height * tile_width / par_width +
                          h * tile_width / par_width + pw) *
                         (par_out_depth * par_width)) +
                        (fi * par_width + wi);

                    ofmap[ofmap_idx] += ofmap_tile[tiled_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

double get_throughput(uint64_t N, double elapsed, double flop_per_elem) {
  return N * flop_per_elem / elapsed * 1e-9;
}

template <typename T>
void RunCpu(std::vector<T> &ifmap, std::vector<T> &weights,
            std::vector<T> &bias, std::vector<T> &ofmap, int height, int width,
            int in_depth, int out_depth) {
  std::cout << "Starting CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  PointwiseConvolutionCpu<T>(ifmap, weights, bias, ofmap, height, width,
                             in_depth, out_depth);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: "
            << get_throughput((uint64_t)height * width * in_depth * out_depth,
                              elapsed.count(), 2) << " GFLOPs" << std::endl;
}

template <typename T>
void RunTiledCpu(std::vector<T> &ifmap, std::vector<T> &weights,
                 std::vector<T> &bias, std::vector<T> &ofmap, int height,
                 int width, int in_depth, int out_depth) {
  max_file_t *max_file = PointwiseConvolution_init();

  auto tile_height = max_get_constant_uint64t(max_file, "TILE_HEIGHT");
  auto tile_width = max_get_constant_uint64t(max_file, "TILE_WIDTH");
  auto tile_in_depth = max_get_constant_uint64t(max_file, "TILE_IN_DEPTH");
  auto tile_out_depth = max_get_constant_uint64t(max_file, "TILE_OUT_DEPTH");

  std::cout << "Starting Tiled CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  PointwiseConvolutionTiledCpu<T>(ifmap, weights, bias, ofmap, height, width,
                                  in_depth, out_depth, tile_height, tile_width,
                                  tile_in_depth, tile_out_depth);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: "
            << get_throughput((uint64_t)height * width * in_depth * out_depth,
                              elapsed.count(), 2) << " GFLOPs" << std::endl;

  max_file_free(max_file);
}

template <typename T>
void RunDfe(std::vector<T> &ifmap, std::vector<T> &weights,
            std::vector<T> &bias, std::vector<T> &ofmap, int height, int width,
            int in_depth, int out_depth) {
  max_file_t *max_file = PointwiseConvolution_init();
  max_engine_t *engine = max_load(max_file, "*");

  auto tile_height = max_get_constant_uint64t(max_file, "TILE_HEIGHT");
  auto tile_width = max_get_constant_uint64t(max_file, "TILE_WIDTH");
  auto tile_in_depth = max_get_constant_uint64t(max_file, "TILE_IN_DEPTH");
  auto tile_out_depth = max_get_constant_uint64t(max_file, "TILE_OUT_DEPTH");
  auto par_width = max_get_constant_uint64t(max_file, "PAR_WIDTH");
  auto par_in_depth = max_get_constant_uint64t(max_file, "PAR_IN_DEPTH");
  auto par_out_depth = max_get_constant_uint64t(max_file, "PAR_OUT_DEPTH");

  std::cout << "Starting DFE ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  PointwiseConvolutionDfe<T>(engine, ifmap, weights, bias, ofmap, height, width,
                             in_depth, out_depth, tile_height, tile_width,
                             tile_in_depth, tile_out_depth, par_width,
                             par_in_depth, par_out_depth);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: "
            << get_throughput((uint64_t)height * width * in_depth * out_depth,
                              elapsed.count(), 2) << " GFLOPs" << std::endl;

  max_unload(engine);
  max_file_free(max_file);
}

typedef float T;

int main(int argc, char *argv[]) {
  srand(42);
  google::InitGoogleLogging(argv[0]);

  const int height = 128;
  const int width = 128;
  const int in_depth = 512;
  const int out_depth = 512;

  std::vector<T> ifmap(in_depth * height * width);
  std::vector<T> weights(out_depth * in_depth);
  std::vector<T> bias(out_depth);
  std::vector<T> golden(out_depth * height * width);
  std::vector<T> tiled_result(out_depth * height * width);
  std::vector<T> dfe_result(out_depth * height * width);

  for (int i = 0; i < (int)ifmap.size(); i++)
    ifmap[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)weights.size(); i++)
    weights[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)bias.size(); i++)
    bias[i] = (T)((double)rand() / RAND_MAX);

  RunCpu<T>(ifmap, weights, bias, golden, height, width, in_depth, out_depth);
  RunTiledCpu<T>(ifmap, weights, bias, tiled_result, height, width, in_depth,
                 out_depth);
  RunDfe<T>(ifmap, weights, bias, dfe_result, height, width, in_depth,
            out_depth);

  for (int i = 0; i < (int)tiled_result.size(); i++)
    CHECK(fabs(golden[i] - tiled_result[i]) < 1e-3)
        << "golden and tiled result should match: " << golden[i] << " "
        << tiled_result[i];

  // for (int i = 0; i < (int)tiled_result.size(); i++)
  //   printf("tiled result[%5d] = %10.6f\n", i, tiled_result[i]);

  // for (int i = 0; i < (int)dfe_result.size(); i++)
  //   printf("dfe result[%5d] = %10.6f\n", i, dfe_result[i]);

  for (int i = 0; i < (int)dfe_result.size(); i++)
    CHECK(fabs(golden[i] - dfe_result[i]) < 1e-3)
        << "golden and dfe result should match at " << i << ": " << golden[i]
        << " " << dfe_result[i];

  return 0;
}
