#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "Maxfiles.h"

DEFINE_int32(height, 66, "Height of the input feature map");
DEFINE_int32(width, 66, "Width of the input fetaure map");
DEFINE_int32(in_depth, 512, "Depth of the input feature map");
DEFINE_int32(out_depth, 512, "Depth of the output feature map");

typedef float T;

double get_throughput(uint64_t N, double elapsed, double flop_per_elem) {
  return N * flop_per_elem / elapsed * 1e-9;
}

int get_num_tiles(int full, int tile) { return (int)ceil((double)full / tile); }

template <typename T>
void DepthwiseSeparableCpu(T *ifmap, T *depthwise_weights, T *pointwise_weights,
                           T *ofmap, int height, int width, int in_depth,
                           int out_depth, int kernel_size) {
  int out_height = height - kernel_size + 1;
  int out_width = width - kernel_size + 1;

  auto depthwise_result = new T[out_height * out_width * in_depth];

  for (int d = 0; d < in_depth; d++) {
    for (int oh = 0; oh < out_height; oh++) {
      for (int ow = 0; ow < out_width; ow++) {
        auto idx = d * out_height * out_width + oh * out_width + ow;
        depthwise_result[idx] = (T)0.0f;

        for (int kh = 0; kh < kernel_size; kh++) {
          for (int kw = 0; kw < kernel_size; kw++) {
            auto ih = oh + kh;
            auto iw = ow + kw;

            if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;

            auto ifmap_idx = d * height * width + ih * width + iw;
            auto weights_idx =
                d * kernel_size * kernel_size + kh * kernel_size + kw;

            depthwise_result[idx] +=
                ifmap[ifmap_idx] * depthwise_weights[weights_idx];
          }
        }
      }
    }
  }

  for (int f = 0; f < out_depth; f++) {
    for (int c = 0; c < in_depth; c++) {
      for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
          auto ofmap_idx = f * out_height * out_width + h * out_width + w;
          auto ifmap_val =
              depthwise_result[c * out_height * out_width + h * out_width + w];
          auto weights_val = pointwise_weights[f * in_depth + c];

          ofmap[ofmap_idx] += ifmap_val * weights_val;
        }
      }
    }
  }
}

template <typename T>
T *PrepareTiledIfmap(T *ifmap, int H, int W, int ID, int OD, int K, int TH,
                     int TW, int TID, int TOD, int PW = 1, int PD = 1) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTID = get_num_tiles(ID, TID);
  auto NTOD = get_num_tiles(ID, TOD);
  auto TIH = TH + K - 1;
  auto TIW = TW + K - 1;
  auto NT = NTH * NTW * NTID * NTOD;
  auto TN = TIH * TIW * TID;
  auto N = NT * TN;
  auto HK = (int)floor((double)K / 2) + 1;

  auto tiled = new T[N];

  for (int tfi = 0; tfi < NTOD; tfi++) {
    for (int tdi = 0; tdi < NTID; tdi++) {
      for (int thi = 0; thi < NTH; thi++) {
        for (int twi = 0; twi < NTW; twi++) {
          auto di = tdi * TID;
          auto hi = thi * TIH - HK * thi;
          auto wi = twi * TIW - HK * twi;
          auto ti = tfi * NTID * NTH * NTW + tdi * NTH * NTW + thi * NTW + twi;
          auto tii = ti * TN;

          for (int dj = 0; dj < TID; dj += PD) {
            for (int hj = 0; hj < TIH; hj++) {
              for (int wj = 0; wj < TIW; wj += PW) {
                for (int pdi = 0; pdi < PD; pdi++) {
                  for (int pwi = 0; pwi < PW; pwi++) {
                    auto pdj = dj / PD;
                    auto pwj = wj / PW;
                    auto tj = ((pdj * TIH * TIW / PW + hj * TIW / PW + pwj) *
                               PW * PD) +
                              (pdi * PW + pwi);
                    auto dii = di + dj + pdi;
                    auto hii = hi + hj;
                    auto wii = wi + wj + pwi;

                    tiled[tii + tj] = (dii >= ID || hii >= H || wii >= W)
                                          ? (T)0.0f
                                          : ifmap[dii * H * W + hii * W + wii];
                  }
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
T *PrepareTiledDepthwiseWeights(T *weights, int H, int W, int ID, int OD, int K,
                                int TH, int TW, int TID, int TOD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTID = get_num_tiles(ID, TID);
  auto NTOD = get_num_tiles(OD, TOD);
  auto NT = NTH * NTW * NTID * NTOD;
  auto TN = TOD * K * K;
  auto N = TN * NT;

  auto tiled = new T[N];

  for (int tf = 0; tf < NTOD; tf++) {
    for (int tc = 0; tc < NTID; tc++) {
      for (int th = 0; th < NTH; th++) {
        for (int tw = 0; tw < NTW; tw++) {

          for (int c = 0; c < TID; c++) {
            for (int k = 0; k < K * K; k++) {
              auto ci = tc * TID + c;
              auto tile_idx =
                  tf * NTID * NTH * NTW + tc * NTH * NTW + th * NTW + tw;
              auto ti = tile_idx * TN + c * K * K + k;

              tiled[ti] = weights[ci * K * K + k];
            }
          }
        }
      }
    }
  }

  return tiled;
}

template <typename T>
T *PrepareTiledPointwiseWeights(T *weights, int out_height, int out_width,
                                int in_depth, int out_depth, int tile_height,
                                int tile_width, int tile_in_depth,
                                int tile_out_depth, int par_in_depth = 1,
                                int par_out_depth = 1) {
  auto num_tiles_height = get_num_tiles(out_height, tile_height);
  auto num_tiles_width = get_num_tiles(out_width, tile_width);
  auto num_tiles_in_depth = get_num_tiles(in_depth, tile_in_depth);
  auto num_tiles_out_depth = get_num_tiles(out_depth, tile_out_depth);
  auto num_tiles = num_tiles_height * num_tiles_width * num_tiles_in_depth *
                   num_tiles_out_depth;
  auto tile_num_elems = tile_in_depth * tile_out_depth;

  T *tiled = new T[num_tiles * tile_num_elems];

  for (int tf = 0; tf < num_tiles_out_depth; tf++) {
    for (int tc = 0; tc < num_tiles_in_depth; tc++) {
      for (int th = 0; th < num_tiles_height; th++) {
        for (int tw = 0; tw < num_tiles_width; tw++) {
          auto sf = tf * tile_out_depth;
          auto sc = tc * tile_in_depth;
          auto tile_idx =
              (tf * num_tiles_in_depth * num_tiles_height * num_tiles_width +
               tc * num_tiles_height * num_tiles_width + th * num_tiles_width +
               tw);

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

                  tiled[tile_idx * tile_num_elems + tiled_idx] =
                      (rc >= in_depth || rf >= out_depth)
                          ? (T)0.0f
                          : weights[weights_idx];
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
void DepthwiseSeparableTiledCpu(T *ifmap, T *depthwise_weights,
                                T *pointwise_weights, T *ofmap, int H, int W,
                                int ID, int OD, int K, int TH, int TW, int TID,
                                int TOD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTID = get_num_tiles(ID, TID);
  auto NTOD = get_num_tiles(OD, TOD);
  auto NT = NTH * NTW * NTID * NTOD;
  auto TIH = TH + K - 1;
  auto TIW = TW + K - 1;
  auto TIN = TIH * TIW * TID;
  auto TDWN = TID * K * K;
  auto TPWN = TID * TOD;
  auto TON = TH * TW * TOD;

  auto tiled_ifmap =
      PrepareTiledIfmap<T>(ifmap, H, W, ID, OD, K, TH, TW, TID, TOD);
  auto tiled_depthwise_weights = PrepareTiledDepthwiseWeights<T>(
      depthwise_weights, H, W, ID, OD, K, TH, TW, TID, TOD);
  auto tiled_pointwise_weights = PrepareTiledPointwiseWeights<T>(
      pointwise_weights, OH, OW, ID, OD, TH, TW, TID, TOD);
  auto tiled_ofmap = new T[NT * TON];

  for (int tf = 0; tf < NTOD; tf++) {
    for (int tc = 0; tc < NTID; tc++) {
      for (int th = 0; th < NTH; th++) {
        for (int tw = 0; tw < NTW; tw++) {
          auto tile_idx =
              tf * NTID * NTH * NTW + tc * NTH * NTW + th * NTW + tw;
          auto ifmap_ptr = &tiled_ifmap[tile_idx * TIN];
          auto depthwise_weights_ptr =
              &tiled_depthwise_weights[tile_idx * TDWN];
          auto pointwise_weights_ptr =
              &tiled_pointwise_weights[tile_idx * TPWN];
          auto ofmap_ptr = &tiled_ofmap[tile_idx * TON];

          DepthwiseSeparableCpu<T>(ifmap_ptr, depthwise_weights_ptr,
                                   pointwise_weights_ptr, ofmap_ptr, TIH, TIW,
                                   TID, TOD, K);

          for (int fi = 0; fi < TOD; fi++) {
            for (int hi = 0; hi < TH; hi++) {
              for (int wi = 0; wi < TW; wi++) {
                ofmap[(tf * TOD + fi) * OH * OW + (th * TH + hi) * OW +
                      (tw * TW + wi)] += ofmap_ptr[fi * TH * TW + hi * TW + wi];
              }
            }
          }
        }
      }
    }
  }

  delete tiled_ifmap;
  delete tiled_depthwise_weights;
  delete tiled_pointwise_weights;
  delete tiled_ofmap;
}

template <typename T>
void DepthwiseSeparableDfe(max_engine_t *engine, T *ifmap, T *depthwise_weights,
                           T *pointwise_weights, T *ofmap, int H, int W, int ID,
                           int OD, int K, int TH, int TW, int TID, int TOD,
                           int PW, int PID, int POD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTID = get_num_tiles(ID, TID);
  auto NTOD = get_num_tiles(OD, TOD);
  auto NT = NTH * NTW * NTID * NTOD;
  auto TON = TH * TW * TOD;

  auto tiled_ifmap =
      PrepareTiledIfmap<T>(ifmap, H, W, ID, OD, K, TH, TW, TID, TOD, PW, PID);
  auto tiled_depthwise_weights = PrepareTiledDepthwiseWeights<T>(
      depthwise_weights, H, W, ID, OD, K, TH, TW, TID, TOD);
  auto tiled_pointwise_weights = PrepareTiledPointwiseWeights<T>(
      pointwise_weights, OH, OW, ID, OD, TH, TW, TID, TOD, PID, POD);
  auto tiled_ofmap = new T[NT * TON];

  DepthwiseSeparable_actions_t actions;
  actions.param_N = (const int64_t)NT;
  actions.instream_ifmap = (const T *)tiled_ifmap;
  actions.instream_depthwise_weights = (const T *)tiled_depthwise_weights;
  actions.instream_pointwise_weights = (const T *)tiled_pointwise_weights;
  actions.outstream_ofmap = tiled_ofmap;

  auto start = std::chrono::system_clock::now();
  DepthwiseSeparable_run(engine, &actions);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)(H * W * ID * K * K + H * W * ID * OD);
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "core elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;

  for (int tf = 0; tf < NTOD; tf++) {
    for (int tc = 0; tc < NTID; tc++) {
      for (int th = 0; th < NTH; th++) {
        for (int tw = 0; tw < NTW; tw++) {
          for (int fi = 0; fi < TOD; fi += POD) {
            for (int hi = 0; hi < TH; hi++) {
              for (int wi = 0; wi < TW; wi += PW) {
                for (int fj = 0; fj < POD; fj++) {
                  for (int wj = 0; wj < PW; wj++) {
                    auto fii = tf * TOD + fi + fj;
                    auto hii = th * TH + hi;
                    auto wii = tw * TW + wi + wj;
                    auto pfi = fi / POD;
                    auto pwi = wi / PW;
                    auto ti =
                        tf * NTID * NTH * NTW + tc * NTH * NTW + th * NTW + tw;
                    auto tii =
                        ti * TON +
                        (pfi * TH * TW / PW + hi * TW / PW + pwi) * POD * PW +
                        (fj * PW + wj);

                    if (fii >= OD || hii >= OH || wii >= OW) continue;

                    ofmap[fii * OH * OW + hii * OW + wii] += tiled_ofmap[tii];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  delete tiled_ifmap;
  delete tiled_depthwise_weights;
  delete tiled_pointwise_weights;
  delete tiled_ofmap;
}

template <typename T>
void RunCpu(std::vector<T> &ifmap, std::vector<T> &depthwise_weights,
            std::vector<T> &pointwise_weights, std::vector<T> &ofmap,
            int height, int width, int in_depth, int out_depth,
            int kernel_size) {
  std::cout << "Starting CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  DepthwiseSeparableCpu<T>((T *)ifmap.data(), (T *)depthwise_weights.data(),
                           (T *)pointwise_weights.data(), (T *)ofmap.data(),
                           height, width, in_depth, out_depth, kernel_size);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * in_depth *
           (kernel_size * kernel_size + out_depth);
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

template <typename T>
void RunTiledCpu(std::vector<T> &ifmap, std::vector<T> &depthwise_weights,
                 std::vector<T> &pointwise_weights, std::vector<T> &ofmap,
                 int height, int width, int in_depth, int out_depth,
                 int kernel_size, int tile_height, int tile_width,
                 int tile_in_depth, int tile_out_depth) {
  std::cout << "Starting Tiled CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  DepthwiseSeparableTiledCpu<T>(
      (T *)ifmap.data(), (T *)depthwise_weights.data(),
      (T *)pointwise_weights.data(), (T *)ofmap.data(), height, width, in_depth,
      out_depth, kernel_size, tile_height, tile_width, tile_in_depth,
      tile_out_depth);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * in_depth *
           (kernel_size * kernel_size + out_depth);
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

template <typename T>
void RunDfe(max_engine_t *engine, std::vector<T> &ifmap,
            std::vector<T> &depthwise_weights,
            std::vector<T> &pointwise_weights, std::vector<T> &ofmap,
            int height, int width, int in_depth, int out_depth, int kernel_size,
            int tile_height, int tile_width, int tile_in_depth,
            int tile_out_depth, int par_width, int par_in_depth,
            int par_out_depth) {
  std::cout << "Starting DFE ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  DepthwiseSeparableDfe<T>(
      engine, (T *)ifmap.data(), (T *)depthwise_weights.data(),
      (T *)pointwise_weights.data(), (T *)ofmap.data(), height, width, in_depth,
      out_depth, kernel_size, tile_height, tile_width, tile_in_depth,
      tile_out_depth, par_width, par_in_depth, par_out_depth);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * in_depth *
           (kernel_size * kernel_size + out_depth);
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

int main(int argc, char *argv[]) {
  srand(42);
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  max_file_t *max_file = DepthwiseSeparable_init();
  max_engine_t *engine = max_load(max_file, "*");

  auto tile_height = max_get_constant_uint64t(max_file, "TILE_HEIGHT");
  auto tile_width = max_get_constant_uint64t(max_file, "TILE_WIDTH");
  auto tile_in_depth = max_get_constant_uint64t(max_file, "TILE_IN_DEPTH");
  auto tile_out_depth = max_get_constant_uint64t(max_file, "TILE_OUT_DEPTH");
  auto par_width = max_get_constant_uint64t(max_file, "PAR_WIDTH");
  auto par_in_depth = max_get_constant_uint64t(max_file, "PAR_IN_DEPTH");
  auto par_out_depth = max_get_constant_uint64t(max_file, "PAR_OUT_DEPTH");
  auto kernel_size = max_get_constant_uint64t(max_file, "KERNEL_SIZE");

  std::vector<T> ifmap(FLAGS_in_depth * FLAGS_height * FLAGS_width);
  std::vector<T> depthwise_weights(FLAGS_in_depth * kernel_size * kernel_size);
  std::vector<T> pointwise_weights(FLAGS_in_depth * FLAGS_out_depth);

  std::vector<T> golden(FLAGS_out_depth * (FLAGS_height - kernel_size + 1) *
                        (FLAGS_width - kernel_size + 1));
  std::vector<T> tiled_cpu(FLAGS_out_depth * (FLAGS_height - kernel_size + 1) *
                           (FLAGS_width - kernel_size + 1));
  std::vector<T> result(FLAGS_out_depth * (FLAGS_height - kernel_size + 1) *
                        (FLAGS_width - kernel_size + 1));

  for (int i = 0; i < (int)ifmap.size(); i++)
    ifmap[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)depthwise_weights.size(); i++)
    depthwise_weights[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)pointwise_weights.size(); i++)
    pointwise_weights[i] = (T)((double)rand() / RAND_MAX);

  RunCpu<T>(ifmap, depthwise_weights, pointwise_weights, golden, FLAGS_height,
            FLAGS_width, FLAGS_in_depth, FLAGS_out_depth, kernel_size);
  RunTiledCpu<T>(ifmap, depthwise_weights, pointwise_weights, tiled_cpu,
                 FLAGS_height, FLAGS_width, FLAGS_in_depth, FLAGS_out_depth,
                 kernel_size, tile_height, tile_width, tile_in_depth,
                 tile_out_depth);
  for (int i = 0; i < (int)tiled_cpu.size(); i++)
    CHECK(fabs(golden[i] - tiled_cpu[i]) < 1e-2)
        << "golden and tiled result should match at " << i << " : " << golden[i]
        << " " << tiled_cpu[i];

  RunDfe<T>(engine, ifmap, depthwise_weights, pointwise_weights, result,
            FLAGS_height, FLAGS_width, FLAGS_in_depth, FLAGS_out_depth,
            kernel_size, tile_height, tile_width, tile_in_depth, tile_out_depth,
            par_width, par_in_depth, par_out_depth);

  // for (int i = 0; i < (int)golden.size(); i++)
  //   printf("golden[%5d] = %.6f\n", i, golden[i]);
  //
  for (int i = 0; i < (int)result.size(); i++)
    CHECK(fabs(golden[i] - result[i]) < 1e-2)
        << "golden and DFE result should match at " << i << " : " << golden[i]
        << " " << result[i];

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
