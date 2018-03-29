#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "Maxfiles.h"

DEFINE_int32(height, 34, "Height of the input feature map");
DEFINE_int32(width, 34, "Width of the input fetaure map");
DEFINE_int32(depth, 32, "Depth of the input feature map");

typedef float T;

double get_throughput(uint64_t N, double elapsed, double flop_per_elem) {
  return N * flop_per_elem / elapsed * 1e-9;
}

int get_num_tiles(int full, int tile) { return (int)ceil((double)full / tile); }

template <typename T>
void DepthwiseConvolutionCpu(T *ifmap, T *weights, T *bias, T *ofmap,
                             int height, int width, int depth,
                             int kernel_size) {
  int out_height = height - kernel_size + 1;
  int out_width = width - kernel_size + 1;

  for (int oh = 0; oh < out_height; oh++) {
    for (int ow = 0; ow < out_width; ow++) {
      for (int d = 0; d < depth; d++) {
        auto ofmap_idx = d * out_height * out_width + oh * out_width + ow;
        ofmap[ofmap_idx] = bias[d];

        for (int kh = 0; kh < kernel_size; kh++) {
          for (int kw = 0; kw < kernel_size; kw++) {
            auto ih = oh + kh;
            auto iw = ow + kw;

            if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;

            auto ifmap_idx = d * height * width + ih * width + iw;
            auto weights_idx =
                d * kernel_size * kernel_size + kh * kernel_size + kw;

            ofmap[ofmap_idx] += ifmap[ifmap_idx] * weights[weights_idx];
          }
        }
      }
    }
  }
}

template <typename T, int M, int N, int K>
void Matmul(T A[M][K], T B[K][N], T C[M][N]) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      C[m][n] = (T)0.0f;
      for (int k = 0; k < K; k++) {
        C[m][n] += A[m][k] * B[k][n];
      }
    }
  }
}

template <typename T, int M, int N>
void Transpose(T A[M][N], T B[N][M]) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) B[n][m] = A[m][n];
}

template <typename T, int M, int N>
void Transform(T A[M][N], T B[N][N], T R[M][M]) {
  T A_T[N][M];
  T A_B[M][N];
  Transpose<T, M, N>(A, A_T);
  Matmul<T, M, N, N>(A, B, A_B);
  Matmul<T, M, M, N>(A_B, A_T, R);
}

template <typename T>
void DepthwiseConvolutionWinogradCpu(T *ifmap, T *weights, T *bias, T *ofmap,
                                     int height, int width, int depth,
                                     int kernel_size) {
  const int M = 4;
  const int R = 3;
  const int TILE_SIZE = M + R - 1;
  int out_height = height - kernel_size + 1;
  int out_width = width - kernel_size + 1;
  int num_tiles_height = get_num_tiles(out_height, M);
  int num_tiles_width = get_num_tiles(out_width, M);

  auto tiled_ifmap = new T[TILE_SIZE][TILE_SIZE];
  auto tiled_coeff = new T[R][R];
  auto trans_ifmap = new T[TILE_SIZE][TILE_SIZE];
  auto trans_coeff = new T[TILE_SIZE][TILE_SIZE];
  auto eltwise = new T[TILE_SIZE][TILE_SIZE];
  auto trans_ofmap = new T[M][M];

  T B[TILE_SIZE][TILE_SIZE] = {{4, 0, 0, 0, 0, 0},
                               {0, -4, 4, -2, 2, 4},
                               {-5, -4, -4, -1, -1, 0},
                               {0, 1, -1, 2, -2, -5},
                               {1, 1, 1, 1, 1, 0},
                               {0, 0, 0, 0, 0, 1}};
  T G[TILE_SIZE][R] = {{0.25f, 0.0, 0.0},
                       {-1. / 6, -1. / 6, -1. / 6},
                       {-1. / 6, 1. / 6, -1. / 6},
                       {1. / 24, 1. / 12, 1. / 6},
                       {1. / 24, -1. / 12, 1. / 6},
                       {0.0, 0.0, 1.0}};
  T A[TILE_SIZE][M] = {{1, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, -1, 1, -1},
                       {1, 2, 4, 8},
                       {1, -2, 4, -8},
                       {0, 0, 0, 1}};
  T A_T[M][TILE_SIZE];
  T B_T[TILE_SIZE][TILE_SIZE];
  Transpose<T, TILE_SIZE, TILE_SIZE>(B, B_T);
  Transpose<T, TILE_SIZE, M>(A, A_T);

  for (int d = 0; d < depth; d++) {
    for (int kh = 0; kh < kernel_size; kh++)
      for (int kw = 0; kw < kernel_size; kw++)
        tiled_coeff[kh][kw] =
            weights[d * kernel_size * kernel_size + kh * kernel_size + kw];

    for (int th = 0; th < num_tiles_height; th++) {
      for (int tw = 0; tw < num_tiles_width; tw++) {
        auto hi = th * M;
        auto wi = tw * M;

        for (int hj = 0; hj < TILE_SIZE; hj++)
          for (int wj = 0; wj < TILE_SIZE; wj++)
            tiled_ifmap[hj][wj] =
                ifmap[d * height * width + (hi + hj) * width + (wi + wj)];

        Transform<T, TILE_SIZE, TILE_SIZE>(B_T, tiled_ifmap, trans_ifmap);
        Transform<T, TILE_SIZE, R>(G, tiled_coeff, trans_coeff);

        for (int hj = 0; hj < TILE_SIZE; hj++)
          for (int wj = 0; wj < TILE_SIZE; wj++)
            eltwise[hj][wj] = trans_coeff[hj][wj] * trans_ifmap[hj][wj];

        Transform<T, M, TILE_SIZE>(A_T, eltwise, trans_ofmap);

        for (int hj = 0; hj < M; hj++)
          for (int wj = 0; wj < M; wj++)
            ofmap[d * out_height * out_width + (th * M + hj) * out_width +
                  (tw * M + wj)] = trans_ofmap[hj][wj] + bias[d];
      }
    }
  }
}

template <typename T>
T *PrepareTiledIfmap(T *ifmap, int H, int W, int D, int K, int TH, int TW,
                     int TD, int PW = 1, int PD = 1) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTD = get_num_tiles(D, TD);
  auto TIH = TH + K - 1;
  auto TIW = TW + K - 1;
  auto NT = NTH * NTW * NTD;
  auto TN = TIH * TIW * TD;
  auto N = NT * TN;
  auto HK = (int)floor((double)K / 2) + 1;

  auto tiled = new T[N];

  for (int tdi = 0; tdi < NTD; tdi++) {
    for (int thi = 0; thi < NTH; thi++) {
      for (int twi = 0; twi < NTW; twi++) {
        auto di = tdi * TD;
        auto hi = thi * TIH - HK * thi;
        auto wi = twi * TIW - HK * twi;
        auto ti = tdi * NTH * NTW + thi * NTW + twi;
        auto tii = ti * TN;

        for (int dj = 0; dj < TD; dj += PD) {
          for (int hj = 0; hj < TIH; hj++) {
            for (int wj = 0; wj < TIW; wj += PW) {
              for (int pdi = 0; pdi < PD; pdi++) {
                for (int pwi = 0; pwi < PW; pwi++) {
                  auto pdj = dj / PD;
                  auto pwj = wj / PW;
                  auto tj =
                      ((pdj * TIH * TIW / PW + hj * TIW / PW + pwj) * PW * PD) +
                      (pdi * PW + pwi);
                  auto dii = di + dj + pdi;
                  auto hii = hi + hj;
                  auto wii = wi + wj + pwi;

                  tiled[tii + tj] = (dii >= D || wii >= W)
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

  return tiled;
}

template <typename T>
T *PrepareTiledWeights(T *weights, int H, int W, int D, int K, int TH, int TW,
                       int TD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTD = get_num_tiles(D, TD);
  auto NT = NTH * NTW * NTD;
  auto TN = TD * K * K;
  auto N = TN * NT;

  T *tiled = new T[N];

  for (int td = 0; td < NTD; td++) {
    for (int th = 0; th < NTH; th++) {
      for (int tw = 0; tw < NTW; tw++) {

        for (int di = 0; di < TD; di++) {
          for (int k = 0; k < K * K; k++) {
            auto d = td * TD + di;
            auto ti = ((td * NTH * NTW + th * NTW + tw) * TN) + di * K * K + k;

            tiled[ti] = weights[d * K * K + k];
          }
        }
      }
    }
  }

  return tiled;
}

template <typename T>
T *PrepareTiledBias(T *bias, int H, int W, int D, int K, int TH, int TW,
                    int TD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTD = get_num_tiles(D, TD);
  auto NT = NTH * NTW * NTD;
  auto TN = TD;
  auto N = TN * NT;

  T *tiled = new T[N];

  for (int td = 0; td < NTD; td++) {
    for (int th = 0; th < NTH; th++) {
      for (int tw = 0; tw < NTW; tw++) {
        for (int di = 0; di < TD; di++) {
          auto d = td * TD + di;
          auto ti = ((td * NTH * NTW + th * NTW + tw) * TN) + di;

          tiled[ti] = bias[d];
        }
      }
    }
  }

  return tiled;
}

template <typename T>
void DepthwiseConvolutionTiledCpu(T *ifmap, T *weights, T *bias, T *ofmap,
                                  int H, int W, int D, int K, int TH, int TW,
                                  int TD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTD = get_num_tiles(D, TD);
  auto NT = NTH * NTW * NTD;
  auto TIH = TH + K - 1;
  auto TIW = TW + K - 1;
  auto TIN = TIH * TIW * TD;
  auto TWN = TD * K * K;
  auto TBN = TD;
  auto TON = TH * TW * TD;

  T *tiled_ifmap = PrepareTiledIfmap<T>(ifmap, H, W, D, K, TH, TW, TD);
  T *tiled_ofmap = new T[NT * TH * TW * TD];

  for (int td = 0; td < NTD; td++) {
    for (int th = 0; th < NTH; th++) {
      for (int tw = 0; tw < NTW; tw++) {
        T *ifmap_ptr = &tiled_ifmap[(td * NTH * NTW + th * NTW + tw) * TIN];
        T *weights_ptr = &weights[td * TWN];
        T *bias_ptr = &bias[td * TBN];
        T *ofmap_ptr = &tiled_ofmap[(td * NTH * NTW + th * NTW + tw) * TON];

        DepthwiseConvolutionCpu<T>(ifmap_ptr, weights_ptr, bias_ptr, ofmap_ptr,
                                   TIH, TIW, TD, K);

        for (int di = 0; di < TD; di++) {
          for (int hi = 0; hi < TH; hi++) {
            for (int wi = 0; wi < TW; wi++) {
              ofmap[(td * TD + di) * OH * OW + (th * TH + hi) * OW +
                    (tw * TW + wi)] =
                  tiled_ofmap[(td * NTH * NTW + th * NTW + tw) * TON +
                              (di * TH * TW + hi * TW + wi)];
            }
          }
        }
      }
    }
  }

  delete tiled_ifmap;
  delete tiled_ofmap;
}

template <typename T>
void DepthwiseConvolutionTiledWinogradCpu(T *ifmap, T *weights, T *bias,
                                          T *ofmap, int H, int W, int D, int K,
                                          int TH, int TW, int TD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTD = get_num_tiles(D, TD);
  auto NT = NTH * NTW * NTD;
  auto TIH = TH + K - 1;
  auto TIW = TW + K - 1;
  auto TIN = TIH * TIW * TD;
  auto TWN = TD * K * K;
  auto TBN = TD;
  auto TON = TH * TW * TD;

  T *tiled_ifmap = PrepareTiledIfmap<T>(ifmap, H, W, D, K, TH, TW, TD);
  T *tiled_ofmap = new T[NT * TH * TW * TD];

  for (int td = 0; td < NTD; td++) {
    for (int th = 0; th < NTH; th++) {
      for (int tw = 0; tw < NTW; tw++) {
        T *ifmap_ptr = &tiled_ifmap[(td * NTH * NTW + th * NTW + tw) * TIN];
        T *weights_ptr = &weights[td * TWN];
        T *bias_ptr = &bias[td * TBN];
        T *ofmap_ptr = &tiled_ofmap[(td * NTH * NTW + th * NTW + tw) * TON];

        DepthwiseConvolutionWinogradCpu<T>(ifmap_ptr, weights_ptr, bias_ptr,
                                           ofmap_ptr, TIH, TIW, TD, K);

        for (int di = 0; di < TD; di++) {
          for (int hi = 0; hi < TH; hi++) {
            for (int wi = 0; wi < TW; wi++) {
              ofmap[(td * TD + di) * OH * OW + (th * TH + hi) * OW +
                    (tw * TW + wi)] =
                  tiled_ofmap[(td * NTH * NTW + th * NTW + tw) * TON +
                              (di * TH * TW + hi * TW + wi)];
            }
          }
        }
      }
    }
  }

  delete tiled_ifmap;
  delete tiled_ofmap;
}

template <typename T>
void DepthwiseConvolutionDfe(max_engine_t *engine, T *ifmap, T *weights,
                             T *bias, T *ofmap, int H, int W, int D, int K,
                             int TH, int TW, int TD, int PW, int PD) {
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTD = get_num_tiles(D, TD);
  auto NT = NTH * NTW * NTD;
  auto TON = TH * TW * TD;

  T *tiled_ifmap = PrepareTiledIfmap<T>(ifmap, H, W, D, K, TH, TW, TD, PW, PD);
  T *tiled_weights = PrepareTiledWeights<T>(weights, H, W, D, K, TH, TW, TD);
  T *tiled_bias = PrepareTiledBias<T>(bias, H, W, D, K, TH, TW, TD);
  T *tiled_ofmap = new T[NT * TON];

  DepthwiseConvolution_actions_t actions;
  actions.param_N = (const int64_t)NT;
  actions.instream_ifmap = (const T *)tiled_ifmap;
  actions.instream_weights = (const T *)tiled_weights;
  actions.instream_bias = (const T *)tiled_bias;
  actions.outstream_ofmap = tiled_ofmap;

  auto start = std::chrono::system_clock::now();
  DepthwiseConvolution_run(engine, &actions);
  auto end = std::chrono::system_clock::now();
  auto N = (uint64_t)H * W * D * K * K;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;

  for (int td = 0; td < NTD; td++) {
    for (int th = 0; th < NTH; th++) {
      for (int tw = 0; tw < NTW; tw++) {
        for (int di = 0; di < TD; di += PD) {
          for (int hi = 0; hi < TH; hi++) {
            for (int wi = 0; wi < TW; wi += PW) {
              for (int dj = 0; dj < PD; dj++) {
                for (int wj = 0; wj < PW; wj++) {
                  auto dii = td * TD + di + dj;
                  auto hii = th * TH + hi;
                  auto wii = tw * TW + wi + wj;
                  auto pdi = di / PD;
                  auto pwi = wi / PW;
                  auto tii =
                      (td * NTH * NTW + th * NTW + tw) * TON +
                      (pdi * TH * TW / PW + hi * TW / PW + pwi) * PD * PW +
                      (dj * PW + wj);

                  if (dii >= D || hii >= H || wii >= W) continue;

                  ofmap[dii * OH * OW + hii * OW + wii] = tiled_ofmap[tii];
                }
              }
            }
          }
        }
      }
    }
  }

  delete tiled_ifmap;
  delete tiled_weights;
  delete tiled_bias;
  delete tiled_ofmap;
}

template <typename T>
void DepthwiseConvolutionWinogradDfe(max_engine_t *engine, T *ifmap, T *weights,
                                     T *bias, T *ofmap, int H, int W, int D,
                                     int K, int TH, int TW, int TD, int PD) {
  const int M = 4;
  auto OH = H - K + 1;
  auto OW = W - K + 1;
  auto NTH = get_num_tiles(OH, TH);
  auto NTW = get_num_tiles(OW, TW);
  auto NTD = get_num_tiles(D, TD);
  auto NT = NTH * NTW * NTD;
  auto TON = TH * TW * TD;

  T *tiled_ifmap = PrepareTiledIfmap<T>(ifmap, H, W, D, K, TH, TW, TD, 1, PD);
  T *tiled_weights = PrepareTiledWeights<T>(weights, H, W, D, K, TH, TW, TD);
  T *tiled_bias = PrepareTiledBias<T>(bias, H, W, D, K, TH, TW, TD);
  T *tiled_ofmap = new T[NT * TON];

  DepthwiseConvolution_actions_t actions;
  actions.param_N = (const int64_t)NT;
  actions.instream_ifmap = (const T *)tiled_ifmap;
  actions.instream_weights = (const T *)tiled_weights;
  actions.instream_bias = (const T *)tiled_bias;
  actions.outstream_ofmap = tiled_ofmap;

  auto start = std::chrono::system_clock::now();
  DepthwiseConvolution_run(engine, &actions);
  auto end = std::chrono::system_clock::now();
  auto N = (uint64_t)H * W * D * K * K;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;

  for (int td = 0; td < NTD; td++) {
    for (int th = 0; th < NTH; th++) {
      for (int tw = 0; tw < NTW; tw++) {
        for (int di = 0; di < TD; di += PD) {
          for (int hi = 0; hi < TH; hi += M) {
            for (int wi = 0; wi < TW; wi += M) {
              for (int dj = 0; dj < PD; dj++) {
                for (int hj = 0; hj < M; hj++) {
                  for (int wj = 0; wj < M; wj++) {
                    auto dii = td * TD + di + dj;
                    auto hii = th * TH + hi + hj;
                    auto wii = tw * TW + wi + wj;
                    auto pdi = di / PD;
                    auto phi = hi / M;
                    auto pwi = wi / M;
                    auto tii = (td * NTH * NTW + th * NTW + tw) * TON +
                               (pdi * TH / M * TW / M + phi * TW / M + pwi) *
                                   PD * M * M +
                               (dj * M * M + hj * M + wj);

                    if (dii >= D || hii >= H || wii >= W) continue;

                    ofmap[dii * OH * OW + hii * OW + wii] = tiled_ofmap[tii];
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
  delete tiled_weights;
  delete tiled_bias;
  delete tiled_ofmap;
}
template <typename T>
void RunCpu(std::vector<T> &ifmap, std::vector<T> &weights,
            std::vector<T> &bias, std::vector<T> &ofmap, int height, int width,
            int depth, int kernel_size) {
  std::cout << "Starting CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  DepthwiseConvolutionCpu<T>((T *)ifmap.data(), (T *)weights.data(),
                             (T *)bias.data(), (T *)ofmap.data(), FLAGS_height,
                             FLAGS_width, FLAGS_depth, kernel_size);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * depth * kernel_size * kernel_size;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

template <typename T>
void RunWinogradCpu(std::vector<T> &ifmap, std::vector<T> &weights,
                    std::vector<T> &bias, std::vector<T> &ofmap, int height,
                    int width, int depth, int kernel_size) {
  std::cout << "Starting Winograd CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  DepthwiseConvolutionWinogradCpu<T>(
      (T *)ifmap.data(), (T *)weights.data(), (T *)bias.data(),
      (T *)ofmap.data(), FLAGS_height, FLAGS_width, FLAGS_depth, kernel_size);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * depth * kernel_size * kernel_size;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

template <typename T>
void RunTiledCpu(std::vector<T> &ifmap, std::vector<T> &weights,
                 std::vector<T> &bias, std::vector<T> &ofmap, int height,
                 int width, int depth, int kernel_size, int tile_height,
                 int tile_width, int tile_depth) {
  std::cout << "Starting Tiled CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  DepthwiseConvolutionTiledCpu<T>((T *)ifmap.data(), (T *)weights.data(),
                                  (T *)bias.data(), (T *)ofmap.data(), height,
                                  width, depth, kernel_size, tile_height,
                                  tile_width, tile_depth);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * depth * kernel_size * kernel_size;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

template <typename T>
void RunTiledWinogradCpu(std::vector<T> &ifmap, std::vector<T> &weights,
                         std::vector<T> &bias, std::vector<T> &ofmap,
                         int height, int width, int depth, int kernel_size,
                         int tile_height, int tile_width, int tile_depth) {
  std::cout << "Starting Tiled Winograd CPU ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  DepthwiseConvolutionTiledWinogradCpu<T>(
      (T *)ifmap.data(), (T *)weights.data(), (T *)bias.data(),
      (T *)ofmap.data(), height, width, depth, kernel_size, tile_height,
      tile_width, tile_depth);
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * depth * kernel_size * kernel_size;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

template <typename T>
void RunDfe(max_engine_t *engine, std::vector<T> &ifmap,
            std::vector<T> &weights, std::vector<T> &bias,
            std::vector<T> &ofmap, int height, int width, int depth,
            int kernel_size, int tile_height, int tile_width, int tile_depth,
            int par_width, int par_depth, bool use_winograd) {
  std::cout << "Starting DFE ..." << std::endl;
  auto start = std::chrono::system_clock::now();
  if (use_winograd) {
    DepthwiseConvolutionWinogradDfe<T>(
        engine, (T *)ifmap.data(), (T *)weights.data(), (T *)bias.data(),
        (T *)ofmap.data(), height, width, depth, kernel_size, tile_height,
        tile_width, tile_depth, par_depth);
  } else {
    DepthwiseConvolutionDfe<T>(engine, (T *)ifmap.data(), (T *)weights.data(),
                               (T *)bias.data(), (T *)ofmap.data(), height,
                               width, depth, kernel_size, tile_height,
                               tile_width, tile_depth, par_width, par_depth);
  }
  auto end = std::chrono::system_clock::now();

  auto N = (uint64_t)height * width * depth * kernel_size * kernel_size;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;
  std::cout << "throughput: " << get_throughput(N, elapsed.count(), 2)
            << " GFLOPs" << std::endl;
}

int main(int argc, char *argv[]) {
  srand(42);
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  max_file_t *max_file = DepthwiseConvolution_init();
  max_engine_t *engine = max_load(max_file, "*");

  auto tile_height = max_get_constant_uint64t(max_file, "TILE_HEIGHT");
  auto tile_width = max_get_constant_uint64t(max_file, "TILE_WIDTH");
  auto tile_depth = max_get_constant_uint64t(max_file, "TILE_DEPTH");
  auto par_width = max_get_constant_uint64t(max_file, "PAR_WIDTH");
  auto par_depth = max_get_constant_uint64t(max_file, "PAR_DEPTH");
  auto kernel_size = max_get_constant_uint64t(max_file, "KERNEL_SIZE");
  auto use_winograd = max_get_constant_uint64t(max_file, "USE_WINOGRAD");

  std::vector<T> ifmap(FLAGS_depth * FLAGS_height * FLAGS_width);
  std::vector<T> weights(FLAGS_depth * kernel_size * kernel_size);
  std::vector<T> bias(FLAGS_depth);

  std::vector<T> golden(FLAGS_depth * (FLAGS_height - kernel_size + 1) *
                        (FLAGS_width - kernel_size + 1));
  std::vector<T> winograd_cpu(FLAGS_depth * (FLAGS_height - kernel_size + 1) *
                              (FLAGS_width - kernel_size + 1));
  std::vector<T> winograd_tiled_cpu(FLAGS_depth *
                                    (FLAGS_height - kernel_size + 1) *
                                    (FLAGS_width - kernel_size + 1));
  std::vector<T> tiled_cpu(FLAGS_depth * (FLAGS_height - kernel_size + 1) *
                           (FLAGS_width - kernel_size + 1));
  std::vector<T> result(FLAGS_depth * (FLAGS_height - kernel_size + 1) *
                        (FLAGS_width - kernel_size + 1));

  for (int i = 0; i < (int)ifmap.size(); i++)
    ifmap[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)weights.size(); i++)
    weights[i] = (T)((double)rand() / RAND_MAX);
  for (int i = 0; i < (int)bias.size(); i++)
    bias[i] = (T)((double)rand() / RAND_MAX);

  RunCpu<T>(ifmap, weights, bias, golden, FLAGS_height, FLAGS_width,
            FLAGS_depth, kernel_size);
  RunWinogradCpu<T>(ifmap, weights, bias, winograd_cpu, FLAGS_height,
                    FLAGS_width, FLAGS_depth, kernel_size);
  for (int i = 0; i < (int)winograd_cpu.size(); i++)
    CHECK(fabs(golden[i] - winograd_cpu[i]) < 1e-3)
        << "golden and winograd result should match at " << i << " : "
        << golden[i] << " " << winograd_cpu[i];

  RunTiledCpu<T>(ifmap, weights, bias, tiled_cpu, FLAGS_height, FLAGS_width,
                 FLAGS_depth, kernel_size, tile_height, tile_width, tile_depth);
  RunTiledWinogradCpu<T>(ifmap, weights, bias, winograd_tiled_cpu, FLAGS_height,
                         FLAGS_width, FLAGS_depth, kernel_size, tile_height,
                         tile_width, tile_depth);
  for (int i = 0; i < (int)tiled_cpu.size(); i++)
    CHECK(fabs(golden[i] - tiled_cpu[i]) < 1e-3)
        << "golden and tiled result should match at " << i << " : " << golden[i]
        << " " << tiled_cpu[i];
  for (int i = 0; i < (int)winograd_tiled_cpu.size(); i++)
    CHECK(fabs(golden[i] - winograd_tiled_cpu[i]) < 1e-3)
        << "golden and winograd tiled result should match at " << i << " : "
        << golden[i] << " " << winograd_tiled_cpu[i];

  RunDfe<T>(engine, ifmap, weights, bias, result, FLAGS_height, FLAGS_width,
            FLAGS_depth, kernel_size, tile_height, tile_width, tile_depth,
            par_width, par_depth, use_winograd == 1);

  // for (int i = 0; i < (int)golden.size(); i++)
  //   printf("golden[%5d] = %.6f\n", i, golden[i]);
  //
  for (int i = 0; i < (int)result.size(); i++)
    CHECK(fabs(golden[i] - result[i]) < 1e-3)
        << "golden and DFE result should match at " << i << " : " << golden[i]
        << " " << result[i];

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
