#ifndef RUNTIME_LAYERS_H
#define RUNTIME_LAYERS_H
/**
 * Layers - implemented software version of various CNN layers.
 */

#include <glog/logging.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

#ifndef NO_DFE
#include "MaxSLiCInterface.h"
#endif

#include "maxdeep/utils.h"

#define DUMP_ARRAY

template <typename T>
void depthwise_separable_conv_layer(T *ifmap, T *coeff, T *ofmap, int H, int W,
                                    int C, int F, int K, int batch_size) {
  T *depth_coeff = coeff;
  T *point_coeff = &coeff[C * K * K];

  int OH = H - K + 1;
  int OW = W - K + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int f = 0; f < F; f++) {
      for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
          int out_idx = b * F * OH * OW + f * OH * OW + oh * OW + ow;
          ofmap[out_idx] = static_cast<T>(0.0f);

          for (int c = 0; c < C; c++) {
            T sum = static_cast<T>(0.0f);

            for (int kh = 0; kh < K; kh++) {
              for (int kw = 0; kw < K; kw++) {
                int ih = oh + kh;
                int iw = ow + kw;

                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                int in_idx = c * H * W + ih * W + iw;
                int dw_idx = c * K * K + kh * K + kw;

                sum += ifmap[in_idx] * depth_coeff[dw_idx];
              }
            }

            int pw_idx = (f * C + c) * K * K;
            ofmap[out_idx] += sum * point_coeff[pw_idx];
          }
        }
      }
    }
  }
}

int GetConvLayerInputDim(int output_dim, int K, int P, int S) {
  // sanity checks
  CHECK_GT(output_dim, 0);
  CHECK_GT(K, 0);
  CHECK_GE(P, 0);
  CHECK_GT(S, 0);

  return (output_dim - 1) * S + K - 2 * P;
}

int GetConvLayerOutputDim(int input_dim, int K, int P, int S) {
  // sanity checks
  CHECK_GT(input_dim, 0);
  CHECK_GT(K, 0);
  CHECK_GE(P, 0);
  CHECK_GT(S, 0);
  CHECK_EQ((input_dim - K + 2 * P) % S, 0);

  return (input_dim - K + 2 * P) / S + 1;
}

/*! An implementation of convolution layer in software.
 *
 * No tiling involved in this implementation.
 */
template <typename T>
void ConvLayerCpu(std::vector<T> &input, std::vector<T> &weights,
                  std::vector<T> &bias, std::vector<T> &output, int H, int W,
                  int C, int F, int K, int P, int S, bool use_bias = true) {
  CHECK_GT(H, 0);
  CHECK_GT(W, 0);
  CHECK_GT(C, 0);
  CHECK_GT(F, 0);
  CHECK_GT(K, 0);
  CHECK_GE(P, 0);
  CHECK_GT(S, 0);

  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);

  for (int f = 0; f < F; f++) {
    for (int oh = 0; oh < OH; oh++) {
      for (int ow = 0; ow < OW; ow++) {
        auto oi = f * OH * OW + oh * OW + ow;

        output[oi] = (use_bias) ? bias[f] : static_cast<T>(0.0f);

        for (int c = 0; c < C; c++) {
          for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
              auto ih = oh * S + kh - P;
              auto iw = ow * S + kw - P;

              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

              auto input_val = input[c * H * W + ih * W + iw];
              auto weights_val =
                  weights[f * C * K * K + c * K * K + kh * K + kw];

#if 0
              printf(
                  "f = %3d c = %3d oh = %3d ow = %3d kh = %3d kw = %3d input = "
                  "% 2d weight = % 2d\n",
                  f, c, oh, ow, kh, kw, input_val, weights_val);
#endif
              output[oi] += input_val * weights_val;
            }
          }
        }
      }
    }
  }
}

/*! Create tiled input for convolution layer.
 *
 * We will add zeros to the tiled result if P > 0.
 *
 * The number of tiles is decided by the output tile size.
 * But the tile side is decided by the input tile
 *
 * \param input the raw input array
 * \param H height of the input fmap (no padding)
 * \param W width of the input fmap (no padding)
 * \param C channels of the input fmap
 * \param K kernel size
 * \param P padding size
 * \param S stride
 * \param T_OH output tile height
 * \param T_OW output tile width
 * \param TC channels in the tile
 * \param PC level of parallelism along channels
 * \param ND number of duplication
 *
 * \return a tiled array
 */
template <typename T>
std::vector<T> CreateConvLayerTiledInput(std::vector<T> &input, int H, int W,
                                         int C, int K, int P, int S, int T_OH,
                                         int T_OW, int TC, int PC = 1,
                                         int ND = 1) {
  // The list of tiles to return.
  std::vector<T> tiled_input;

  // output dimensions
  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);

  // input tile size
  auto TH = GetConvLayerInputDim(T_OH, K, 0, S);
  auto TW = GetConvLayerInputDim(T_OW, K, 0, S);
  auto tile_size = TH * TW * TC;  // decided by input tile size

  // number of tiles
  auto N_TOH = GetNumTiles(OH, T_OH);
  auto N_TOW = GetNumTiles(OW, T_OW);
  auto N_TC = GetNumTiles(C, TC);
  auto N_T = N_TOH * N_TOW * N_TC;
  auto total_size = N_T * tile_size;

  LOG(INFO) << "Tiling input into " << N_TC << " x " << N_TOH << " x " << N_TOW
            << " number of " << TC << " x " << TH << " x " << TW
            << " tiles ...";

  // half kernel
  auto hk = static_cast<int>(std::ceil(K / 2));

  // resize the return tiles
  tiled_input.resize(total_size * ND);

  for (int td = 0; td < ND; td++) {
    for (int tc = 0; tc < N_TC; tc++) {
      for (int t_oh = 0; t_oh < N_TOH; t_oh++) {
        for (int t_ow = 0; t_ow < N_TOW; t_ow++) {
          // single tile

          for (int c = 0; c < TC; c += PC) {
            for (int h = 0; h < TH; h++) {
              for (int w = 0; w < TW; w++) {
                // The inner loops for building parallelised input
                // data.
                for (int pc = 0; pc < PC; pc++) {
                  // index in the tiled result
                  auto base_idx =
                      td * total_size +
                      (tc * N_TOH * N_TOW + t_oh * N_TOW + t_ow) * tile_size;
                  auto dst_idx = ((c / PC) * TH * TW + h * TW + w) * PC + pc;
                  // index in the input data
                  // hi and wi are padded
                  auto src_ci = tc * TC + c + pc;
                  auto src_hi = t_oh * TH + h - 2 * hk * t_oh;
                  auto src_wi = t_ow * TW + w - 2 * hk * t_ow;
                  auto src_idx =
                      src_ci * H * W + (src_hi - P) * W + (src_wi - P);

#if 0
                  printf(
                      "tc = %3d t_oh = %3d t_ow = %3d c = %3d h = %3d w = %3d "
                      "pc "
                      "= %3d src_hi = %3d src_wi = %3d src_idx = %3d\n",
                      tc, t_oh, t_ow, c, h, w, pc, src_hi, src_wi, src_idx);
#endif

                  // input index lies in the padding range
                  if (src_ci >= C || src_hi < P || src_hi >= H + P ||
                      src_wi < P || src_wi >= W + P)
                    tiled_input[base_idx + dst_idx] = static_cast<T>(0.0f);
                  else
                    tiled_input[base_idx + dst_idx] = input[src_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  return tiled_input;
}

/*! Created an array of tiled convolution weights.
 *
 * We can generated duplicated weight tiles by specifying the
 * duplicate parameter.
 */
template <typename T>
std::vector<T> CreateConvLayerTiledWeights(std::vector<T> &weights, int C,
                                           int F, int K, int TC, int TF,
                                           int PC = 1, int PF = 1,
                                           int duplicate = 1) {
  std::vector<T> tiled_weights;

  auto N_TF = GetNumTiles(F, TF);
  auto N_TC = GetNumTiles(C, TC);
  auto N_T = N_TF * N_TC * duplicate;
  auto tile_size = TC * TF * K * K;
  auto total_size = tile_size * N_T;

  LOG(INFO) << "Tiling weights into " << N_TF << " x " << N_TC << " number of "
            << TF << " x " << TC << " x " << K << " x " << K << " tiles ...";

  tiled_weights.resize(total_size);

  for (int tf = 0; tf < N_TF; tf++) {
    for (int tc = 0; tc < N_TC; tc++) {
      // build a single tile

      for (int td = 0; td < duplicate; td++) {
        for (int f = 0; f < TF; f += PF) {
          for (int c = 0; c < TC; c += PC) {
            // considering parallelisation
            for (int pf = 0; pf < PF; pf++) {
              for (int pc = 0; pc < PC; pc++) {
                for (int k = 0; k < K * K; k++) {
                  auto src_fi = tf * TF + f + pf;
                  auto src_ci = tc * TC + c + pc;
                  if (src_fi >= F || src_ci >= C) continue;

                  auto src_idx = src_fi * C * K * K + src_ci * K * K + k;
                  auto dst_idx =
                      ((((((f / PF) * (TC / PC) + (c / PC)) * PC * PF) +
                         (pf * PC + pc)) *
                        K * K) +
                       k);
                  auto base_idx =
                      ((tf * N_TC + tc) * duplicate + td) * tile_size;

                  tiled_weights[base_idx + dst_idx] = weights[src_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  return tiled_weights;
}

template <typename T>
std::vector<T> TransformConvLayerTiledOutput(std::vector<T> &tiled_output,
                                             int OH, int OW, int F, int T_OH,
                                             int T_OW, int T_F, int P_F = 1,
                                             int N_TC = 1) {
  std::vector<T> output;

  auto N_TOH = GetNumTiles(OH, T_OH);
  auto N_TOW = GetNumTiles(OW, T_OW);
  auto N_TF = GetNumTiles(F, T_F);
  auto N_T = N_TOH * N_TOW * N_TF;
  auto tile_size = T_OH * T_OW * T_F;
  auto total_size = N_T * tile_size;

  LOG(INFO) << "Merging outputs from " << N_TF << " x " << N_TOH << " x "
            << N_TOW << " number of " << T_F << " x " << T_OH << " x " << T_OW
            << " tiles ...";

  output.resize(total_size);

  for (int tf = 0; tf < N_TF; tf++) {
    for (int tc = 0; tc < N_TC; tc++) {
      for (int t_oh = 0; t_oh < N_TOH; t_oh++) {
        for (int t_ow = 0; t_ow < N_TOW; t_ow++) {
          // inside a tile
          for (int f = 0; f < T_F; f += P_F) {
            for (int oh = 0; oh < T_OH; oh++) {
              for (int ow = 0; ow < T_OW; ow++) {
                for (int pf = 0; pf < P_F; pf++) {
                  auto src_idx =
                      ((f / P_F) * T_OH * T_OW + oh * T_OW + ow) * P_F + pf;
                  auto dst_fi = tf * T_F + f + pf;
                  auto dst_hi = t_oh * T_OH + oh;
                  auto dst_wi = t_ow * T_OW + ow;
                  if (dst_fi >= F || dst_hi >= OH || dst_wi >= OW) continue;

                  auto dst_idx = dst_fi * OH * OW + dst_hi * OW + dst_wi;
                  auto base_idx = (tf * N_TC * N_TOH * N_TOW +
                                   tc * N_TOH * N_TOW + t_oh * N_TOW + t_ow) *
                                  tile_size;

                  output[dst_idx] += tiled_output[base_idx + src_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  return output;
}

#ifndef NO_DFE
/*! Run convolution layer on DFE.
 *
 * We will do tiling within this function.
 * Design-specific classes/functions should be specified through
 * template parameters.
 */
template <typename T, typename dfe_actions_t, typename dfe_run_fn>
void ConvLayerDfe(std::vector<T> &input, std::vector<T> &weights,
                  std::vector<T> &bias, std::vector<T> &output, int H, int W,
                  int C, int F, int K, int P, int S, max_file_t *max_file,
                  max_engine_t *engine) {
  // get constants from the max_file
  auto DFE_TH = max_get_constant_uint64t(max_file, "conv_H");
  auto DFE_TW = max_get_constant_uint64t(max_file, "conv_W");
  auto DFE_TOH = GetConvLayerOutputDim(DFE_TH, K, 0, S);
  auto DFE_TOW = GetConvLayerOutputDim(DFE_TW, K, 0, S);
  auto DFE_TC = max_get_constant_uint64t(max_file, "conv_C");
  auto DFE_TF = max_get_constant_uint64t(max_file, "conv_F");
  auto DFE_PC = max_get_constant_uint64t(max_file, "conv_PC");
  auto DFE_PF = max_get_constant_uint64t(max_file, "conv_PF");
  auto DFE_K = max_get_constant_uint64t(max_file, "conv_K");
  auto DFE_USE_DRAM = max_get_constant_uint64t(max_file, "USE_DRAM") == 1;

  // check whether the given convolution layer can be placed on DFE
  CHECK_EQ(static_cast<int>(DFE_K), K);
  CHECK(!DFE_USE_DRAM) << "We don't support DRAM mode in software yet";

  // alias
  auto T_OH = DFE_TOH;
  auto T_OW = DFE_TOW;
  auto TC = DFE_TC;
  auto TF = DFE_TF;

  // output configuration
  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);

  // get number of total tiles
  auto N_TOH = GetNumTiles(OH, T_OH);
  auto N_TOW = GetNumTiles(OW, T_OW);
  auto N_TC = GetNumTiles(C, TC);
  auto N_TF = GetNumTiles(F, TF);
  auto N_T = N_TOH * N_TOW * N_TC * N_TF;

  // output size
  auto output_tile_size = T_OH * T_OW * TF;

  // perform tiling
  // duplicate input with N_TF times
  auto tiled_input = CreateConvLayerTiledInput<T>(input, H, W, C, K, P, S, T_OH,
                                                  T_OW, TC, DFE_PC, N_TF);
  auto tiled_weights = CreateConvLayerTiledWeights<T>(
      weights, C, F, K, TC, TF, DFE_PC, DFE_PF, N_TOH * N_TOW);

#ifdef DUMP_ARRAY
  DumpArray<T>("input.txt", input.data(), input.size());
  DumpArray<T>("tiled_input.txt", tiled_input.data(), tiled_input.size());
  DumpArray<T>("weights.txt", weights.data(), weights.size());
  DumpArray<T>("tiled_weights.txt", tiled_weights.data(), tiled_weights.size());
#endif

  // this tiled output should be further reduced
  std::vector<T> tiled_output(N_T * output_tile_size);

  // create actions for DFE call
  dfe_actions_t actions;
  actions.param_batch_size = N_T;
  actions.instream_ifmap = tiled_input.data();
  actions.instream_coeff_0 = tiled_weights.data();
  actions.outstream_ofmap = reinterpret_cast<T *>(tiled_output.data());

  LOG(INFO) << "Running " << N_T << " convolution on DFE ...";
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  dfe_run_fn().run(engine, &actions);
  end = std::chrono::system_clock::now();
  LOG(INFO) << "Done";

  // transform the the final output
  output = TransformConvLayerTiledOutput<T>(tiled_output, OH, OW, F, T_OH, T_OW,
                                            TF, DFE_PF, N_TC);

  std::chrono::duration<double> elapsed_seconds = end - start;
  LOG(INFO) << "elapsed time: " << elapsed_seconds.count() << "s";
  auto gflops =
      (2 * OH * OW * C * F * K * K) / (elapsed_seconds.count()) * 1e-9;
  LOG(INFO) << "GFLOPS: " << gflops;
}
#endif
#endif
