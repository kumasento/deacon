#ifndef RUNTIME_LAYERS_H
#define RUNTIME_LAYERS_H
/**
 * Layers - implemented software version of various CNN layers.
 */

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

#ifndef NO_DFE
#include "MaxSLiCInterface.h"
#endif

#include "maxdeep/types.h"
#include "maxdeep/utils.h"

#define DUMP_ARRAY
#define WINO_PAD 2
#define WINO_TILE_SIZE 4

// #define TRACE

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

/*! An implementation of convolution layer in software.
 *
 * No tiling involved in this implementation.
 */
template <typename T>
void ConvLayerCpu(std::vector<T> &input, std::vector<T> &weights,
                  std::vector<T> &bias, std::vector<T> &output, int H, int W,
                  int C, int F, int K, int P, int S, bool use_bias = true,
                  bool use_fixed_point = false, int num_frac_bits = 0) {
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
              auto weight_val =
                  weights[f * C * K * K + c * K * K + kh * K + kw];

#ifdef TRACE
              printf("f = %d c = %d oh = %d ow = %d kh = %d kw = %d: ", f, c,
                     oh, ow, kh, kw);
              printf(
                  "input = %10.6f weight = %10.6f\n",
                  ConvertToFloat<T>(input_val, use_fixed_point, num_frac_bits),
                  ConvertToFloat<T>(weight_val, use_fixed_point,
                                    num_frac_bits));

#endif

              if (!use_fixed_point) {
                output[oi] += (input_val * weight_val);
              } else {
                auto mul_val =
                    FixedPointMul<T>(input_val, weight_val, num_frac_bits);

#ifdef TRACE
                printf(
                    "output[%d] = %10.6f + %10.6f = ", oi,
                    ConvertToFloat<T>(output[oi], use_fixed_point,
                                      num_frac_bits),
                    ConvertToFloat<T>(mul_val, use_fixed_point, num_frac_bits));
#endif

                output[oi] = FixedPointAdd<T>(output[oi], mul_val);
#ifdef TRACE
                printf("%10.6f\n",
                       ConvertToFloat<T>(output[oi], use_fixed_point,
                                         num_frac_bits));
#endif
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void ConvLayerCpuBatched(std::vector<T> &input, std::vector<T> &weights,
                         std::vector<T> &bias, std::vector<T> &output, int N,
                         int H, int W, int C, int F, int K, int P, int S,
                         bool use_bias = true, bool use_fixed_point = false,
                         int num_frac_bits = 0) {
  CHECK_GT(N, 0);
  CHECK_GT(H, 0);
  CHECK_GT(W, 0);
  CHECK_GT(C, 0);
  CHECK_GT(F, 0);
  CHECK_GT(K, 0);
  CHECK_GE(P, 0);
  CHECK_GT(S, 0);

  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);

  for (int i = 0; i < N; i++) {
    uint64_t output_offset = (uint64_t)i * (OH * OW * F);
    uint64_t input_offset = (uint64_t)i * (H * W * C);
    for (int f = 0; f < F; f++) {
      for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
          auto oi = output_offset + f * OH * OW + oh * OW + ow;

          output[oi] = (use_bias) ? bias[f] : static_cast<T>(0.0f);

          for (int c = 0; c < C; c++) {
            for (int kh = 0; kh < K; kh++) {
              for (int kw = 0; kw < K; kw++) {
                auto ih = oh * S + kh - P;
                auto iw = ow * S + kw - P;

                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                auto input_val = input[input_offset + c * H * W + ih * W + iw];
                auto weight_val =
                    weights[f * C * K * K + c * K * K + kh * K + kw];

#ifdef TRACE
                printf("f = %d c = %d oh = %d ow = %d kh = %d kw = %d: ", f, c,
                       oh, ow, kh, kw);
                printf("input = %10.6f weight = %10.6f\n",
                       ConvertToFloat<T>(input_val, use_fixed_point,
                                         num_frac_bits),
                       ConvertToFloat<T>(weight_val, use_fixed_point,
                                         num_frac_bits));

#endif

                if (!use_fixed_point) {
                  output[oi] += (input_val * weight_val);
                } else {
                  auto mul_val =
                      FixedPointMul<T>(input_val, weight_val, num_frac_bits);

#ifdef TRACE
                  printf("output[%d] = %10.6f + %10.6f = ", oi,
                         ConvertToFloat<T>(output[oi], use_fixed_point,
                                           num_frac_bits),
                         ConvertToFloat<T>(mul_val, use_fixed_point,
                                           num_frac_bits));
#endif

                  output[oi] = FixedPointAdd<T>(output[oi], mul_val);
#ifdef TRACE
                  printf("%10.6f\n",
                         ConvertToFloat<T>(output[oi], use_fixed_point,
                                           num_frac_bits));
#endif
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void ConvLayerCpuWinograd(std::vector<T> &input, std::vector<T> &weights,
                          std::vector<T> &bias, std::vector<T> &output, int H,
                          int W, int C, int F, int K, int P, int S, int R,
                          bool use_bias = true, bool use_fixed_point = false,
                          int num_frac_bits = 0) {
  CHECK_GT(H, 0);
  CHECK_GT(W, 0);
  CHECK_GT(C, 0);
  CHECK_GT(F, 0);
  CHECK_GT(K, 0);
  CHECK_GE(P, 0);
  CHECK_GT(S, 0);

  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);
  auto M = R - K + 1;

  for (int f = 0; f < F; f++) {
    for (int oh = 0; oh < OH; oh += M) {
      for (int ow = 0; ow < OW; ow += M) {
        for (int c = 0; c < C; c++) {
          std::vector<T> inputs_tile(R * R);
          std::vector<T> weights_tile(K * K);
          std::vector<T> outputs_tile(R * R);

          // prepare inputs tile
          for (int kh = 0; kh < R; kh++) {
            for (int kw = 0; kw < R; kw++) {
              auto ih = oh + kh - P;
              auto iw = ow + kw - P;

              inputs_tile[kh * R + kw] =
                  (ih < 0 || ih >= H || iw < 0 || iw >= W)
                      ? static_cast<T>(0)
                      : input[c * H * W + ih * W + iw];
            }
          }

          // prepare weights tile
          for (int kh = 0; kh < K; kh++)
            for (int kw = 0; kw < K; kw++)
              weights_tile[kh * K + kw] =
                  weights[f * C * K * K + c * K * K + kh * K + kw];

          // transform
          auto inputs_wino = WinogradInputTransform<T>(
              inputs_tile, R, use_fixed_point, num_frac_bits);
          auto weights_wino = WinogradWeightsTransform<T>(
              weights_tile, K, R, use_fixed_point, num_frac_bits);

          for (int k = 0; k < R * R; k++)
            outputs_tile[k] = FixedPointMul<T>(inputs_wino[k], weights_wino[k],
                                               num_frac_bits);

          auto outputs_wino = WinogradOutputTransform<T>(
              outputs_tile, R, M, use_fixed_point, num_frac_bits);

          for (int kh = 0; kh < M; kh++) {
            for (int kw = 0; kw < M; kw++) {
              auto oi = f * OH * OW + (oh + kh) * OW + (ow + kw);
              output[oi] =
                  FixedPointAdd<T>(output[oi], outputs_wino[kh * M + kw]);
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
template <typename T, bool use_wino = false>
std::vector<T> CreateConvLayerTiledInput(std::vector<T> &input, int H, int W,
                                         int C, int K, int P, int S, int T_OH,
                                         int T_OW, int TC, int PC = 1,
                                         int ND = 1) {
  // The list of tiles to return.
  std::vector<T> tiled_input;

  // output dimensions
  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);

  auto PW = use_wino ? WINO_TILE_SIZE : 1;
  auto PH = use_wino ? WINO_TILE_SIZE : 1;

  auto top_left_pad = use_wino ? WINO_PAD : 0;

  // input tile size
  auto TH = GetConvLayerInputDim(T_OH, K, 0, S);
  auto TW = GetConvLayerInputDim(T_OW, K, 0, S);
  if (use_wino) {
    TH += WINO_PAD;
    TW += WINO_PAD;
  }

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
            for (int h = 0; h < TH; h += PH) {
              for (int w = 0; w < TW; w += PW) {
                // The inner loops for building parallelised input
                // data.
                for (int pc = 0; pc < PC; pc++) {
                  for (int ph = 0; ph < PH; ph++) {
                    for (int pw = 0; pw < PW; pw++) {
                      // index in the tiled result
                      auto base_idx = td * total_size + (tc * N_TOH * N_TOW +
                                                         t_oh * N_TOW + t_ow) *
                                                            tile_size;
                      auto dst_idx =
                          (((c / PC) * (TH / PH) * TW + (h / PH) * TW + w) *
                           PC * PH) +
                          (pc * PH * PW + ph * PW + pw);
                      // index in the input data
                      // hi and wi are padded
                      auto src_ci = tc * TC + c + pc;
                      auto src_hi = t_oh * TH + h + ph - 2 * hk * t_oh -
                                    (t_oh + 1) * top_left_pad;
                      auto src_wi = t_ow * TW + w + pw - 2 * hk * t_ow -
                                    (t_ow + 1) * top_left_pad;
                      auto src_idx =
                          src_ci * H * W + (src_hi - P) * W + (src_wi - P);

#ifdef TRACE
                      if (c == 0 && h == 0)
                        printf(
                            "tc = %3d t_oh = %3d t_ow = %3d c = %3d h = %3d w "
                            "= "
                            "%3d pc = %3d ph = %3d pw = %3d src_hi = %3d "
                            "src_wi = %3d "
                            "src_idx = "
                            "%3d\n",
                            tc, t_oh, t_ow, c, h, w, pc, ph, pw, src_hi, src_wi,
                            src_idx);
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
std::vector<T> CreateConvLayerTiledWeights(
    std::vector<T> &weights, int C, int F, int K, int TC, int TF, int PC = 1,
    int PF = 1, bool winograd_coeff_offline = false,
    bool use_fixed_point = false, int num_frac_bits = 0, int duplicate = 1) {
  std::vector<T> tiled_weights;

  auto N_TF = GetNumTiles(F, TF);
  auto N_TC = GetNumTiles(C, TC);
  auto N_T = N_TF * N_TC * duplicate;
  auto WINO_R = WINO_TILE_SIZE + K - 1;
  auto KERN_SIZE = winograd_coeff_offline ? WINO_R : K;
  auto tile_size = TC * TF * KERN_SIZE * KERN_SIZE;
  auto total_size = tile_size * N_T;

  LOG(INFO) << "Tiling weights into " << N_TF << " x " << N_TC << " number of "
            << TF << " x " << TC << " x " << KERN_SIZE << " x " << KERN_SIZE
            << " tiles ...";

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
                auto src_fi = tf * TF + f + pf;
                auto src_ci = tc * TC + c + pc;
                if (src_fi >= F || src_ci >= C) continue;

                if (winograd_coeff_offline) {
                  std::vector<T> kern(K * K);

                  // prepare a single kernel
                  for (int k = 0; k < K * K; k++) {
                    auto src_idx = src_fi * C * K * K + src_ci * K * K + k;
                    kern[k] = weights[src_idx];
                  }

                  auto trans_weights = WinogradWeightsTransform<T>(
                      kern, K, WINO_R, use_fixed_point, num_frac_bits);

                  for (int k = 0; k < WINO_R * WINO_R; k++) {
                    auto dst_idx =
                        ((((((f / PF) * (TC / PC) + (c / PC)) * PC * PF) +
                           (pf * PC + pc)) *
                          WINO_R * WINO_R) +
                         k);
                    auto base_idx =
                        ((tf * N_TC + tc) * duplicate + td) * tile_size;

                    // TODO(vince): implement coefficient offline transform
                    tiled_weights[base_idx + dst_idx] = trans_weights[k];
                  }

                } else {
                  for (int k = 0; k < K * K; k++) {
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
  }

  return tiled_weights;
}

template <typename T>
std::vector<T> TransformConvLayerTiledOutput(
    std::vector<T> &tiled_output, int OH, int OW, int F, int T_OH, int T_OW,
    int T_F, int P_F = 1, int P_OH = 1, int P_OW = 1, int N_TC = 1,
    bool use_fixed_point = false, int num_frac_bits = 0) {
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
            for (int oh = 0; oh < T_OH; oh += P_OH) {
              for (int ow = 0; ow < T_OW; ow += P_OW) {
                for (int pf = 0; pf < P_F; pf++) {
                  for (int p_oh = 0; p_oh < P_OH; p_oh++) {
                    for (int p_ow = 0; p_ow < P_OW; p_ow++) {
                      auto src_idx =
                          (((f / P_F) * (T_OH / P_OH) * (T_OW / P_OW)) +
                           ((oh / P_OH) * (T_OW / P_OW) + (ow / P_OW))) *
                              P_F * P_OH * P_OW +
                          (pf * P_OH * P_OW + p_oh * P_OW + p_ow);

                      auto dst_fi = tf * T_F + f + pf;
                      auto dst_hi = t_oh * T_OH + oh + p_oh;
                      auto dst_wi = t_ow * T_OW + ow + p_ow;

                      if (dst_fi >= F || dst_hi >= OH || dst_wi >= OW) continue;

                      auto dst_idx = dst_fi * OH * OW + dst_hi * OW + dst_wi;
                      auto base_idx =
                          (tf * N_TC * N_TOH * N_TOW + tc * N_TOH * N_TOW +
                           t_oh * N_TOW + t_ow) *
                          tile_size;

                      if (!use_fixed_point)
                        output[dst_idx] += tiled_output[base_idx + src_idx];
                      else
                        output[dst_idx] = FixedPointAdd<T>(
                            output[dst_idx], tiled_output[base_idx + src_idx]);
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

  return output;
}

#ifndef NO_DFE

template <typename T, typename DfeT>
size_t WriteDRAM(std::vector<T> &arr, size_t base_addr, max_engine_t *engine) {
  typename DfeT::dram_write_actions_t dram_write_actions;

  dram_write_actions.param_start_bytes = base_addr;
  dram_write_actions.param_size_bytes = arr.size() * sizeof(T);
  dram_write_actions.instream_fromcpu =
      reinterpret_cast<const uint8_t *>(arr.data());

  LOG(INFO) << std::showbase << std::setfill('0') << std::setw(8) << std::hex
            << "Writing DRAM to " << base_addr << " with size "
            << arr.size() * sizeof(T);

  DfeT::WriteDRAM(engine, &dram_write_actions);

  return base_addr + arr.size() * sizeof(T);
}

template <typename T, typename DfeT>
size_t ReadDRAM(std::vector<T> &arr, size_t base_addr, max_engine_t *engine) {
  typename DfeT::dram_read_actions_t dram_read_actions;

  dram_read_actions.param_start_bytes = base_addr;
  dram_read_actions.param_size_bytes = arr.size() * sizeof(T);
  dram_read_actions.outstream_tocpu = reinterpret_cast<uint8_t *>(arr.data());

  LOG(INFO) << std::showbase << std::setfill('0') << std::setw(8) << std::hex
            << "Reading DRAM from " << base_addr << " with size "
            << arr.size() * sizeof(T);

  DfeT::ReadDRAM(engine, &dram_read_actions);

  return base_addr + arr.size() * sizeof(T);
}

template <typename T>
double **SplitCoeffAndAssign(double **ptr, T *data,
                             const ConvLayerParameters &cp) {
  uint64_t cols = (uint64_t)std::ceil(static_cast<double>(cp.C) / cp.dfe.PC);
  uint64_t fmem_size =
      cols * (uint64_t)std::ceil(static_cast<double>(cp.F) / cp.dfe.PF);

  bool convert = (!std::is_same<T, float>::value);

  for (uint64_t pf = 0; pf < cp.dfe.PF; ++pf)
    for (uint64_t pc = 0; pc < cp.dfe.PC; ++pc)
      for (int kx = 0; kx < cp.K; ++kx)
        for (int ky = 0; ky < cp.K; ++ky) {
          double *arr = (double *)malloc(sizeof(double) * fmem_size);

          for (int f = 0; f < cp.F; f += cp.dfe.PF)
            for (int c = 0; c < cp.C; c += cp.dfe.PC) {
              T value = data[(f + pf) * (cp.C * cp.K * cp.K) +
                             (c + pc) * (cp.K * cp.K) + (kx * cp.K) + ky];
              auto idx = f / cp.dfe.PF * cols + c / cp.dfe.PC;
              if (!convert)
                arr[idx] = value;
              else
                arr[idx] = FixedToFloat<T>(value, cp.dfe.num_frac_bits);
              // LOG(INFO) << "idx = " << idx << " arr[idx] = " << arr[idx]
              //           << '\n';
            }

          *ptr = (double *)arr;
          ++ptr;
        }

  return ptr;
}

/*! Run convolution layer on DFE.
 *
 * We will do tiling within this function.
 * Design-specific classes/functions should be specified through
 * template parameters.
 */
template <typename T, typename DfeT>
void ConvLayerDfe(std::vector<T> &input, std::vector<T> &weights,
                  std::vector<T> &bias, std::vector<T> &output, int H, int W,
                  int C, int F, int K, int P, int S, max_file_t *max_file,
                  max_engine_t *engine, bool use_fixed_point = false,
                  int num_frac_bits = 0) {
  // get constants from the max_file
  // TODO: make sure to replace the conv with their name.
  auto DFE_TH = max_get_constant_uint64t(max_file, "conv_H");
  auto DFE_TW = max_get_constant_uint64t(max_file, "conv_W");
  auto DFE_TOH = GetConvLayerOutputDim(DFE_TH, K, 0, S);
  auto DFE_TOW = GetConvLayerOutputDim(DFE_TW, K, 0, S);
  auto DFE_TC = max_get_constant_uint64t(max_file, "conv_C");
  auto DFE_TF = max_get_constant_uint64t(max_file, "conv_F");
  auto DFE_PC = max_get_constant_uint64t(max_file, "conv_PC");
  auto DFE_PF = max_get_constant_uint64t(max_file, "conv_PF");
  auto DFE_PK = max_get_constant_uint64t(max_file, "conv_PK");
  auto DFE_K = max_get_constant_uint64t(max_file, "conv_K");
  auto DFE_WINO_COEFF_OFFLINE =
      max_get_constant_uint64t(max_file, "WINO_COEFF_OFFLINE") == 1;
  auto DFE_COEFF_ON_CHIP =
      max_get_constant_uint64t(max_file, "conv_COEFF_ON_CHIP");
#ifdef USE_WINO
  constexpr bool DFE_USE_WINO = true;
  auto DFE_WINO_M = max_get_constant_uint64t(max_file, "WINO_M");
#else
  constexpr bool DFE_USE_WINO = false;
  auto DFE_WINO_M = 1;
#endif
  auto DFE_POH = DFE_USE_WINO ? DFE_WINO_M : 1;
  auto DFE_POW = DFE_USE_WINO ? DFE_WINO_M : 1;

  // check whether the given convolution layer can be placed on DFE
  CHECK_EQ(static_cast<int>(DFE_K), K);
  // auto DFE_USE_DRAM = max_get_constant_uint64t(max_file, "USE_DRAM") == 1;
  // CHECK(!DFE_USE_DRAM) << "We don't support DRAM mode in software yet";

  // alias
  auto T_OH = DFE_TOH;
  auto T_OW = DFE_TOW;
  auto TC = DFE_TC;
  auto TF = DFE_TF;

  // output configuration
  auto OH = GetConvLayerOutputDim(H, K, P, S);
  auto OW = GetConvLayerOutputDim(W, K, P, S);

  ConvLayerParameters cp{C, F, K, P, S, DFE_PF, DFE_PC, DFE_PK, num_frac_bits};
  cp.dfe = DfeConvLayerParameters::get(max_file, "conv");

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
  auto tiled_input = CreateConvLayerTiledInput<T, DFE_USE_WINO>(
      input, H, W, C, K, P, S, T_OH, T_OW, TC, DFE_PC, N_TF);
  LOG(INFO) << "Tiled input size: " << tiled_input.size();

  auto tiled_weights = CreateConvLayerTiledWeights<T>(
      weights, C, F, K, TC, TF, DFE_PC, DFE_PF, DFE_WINO_COEFF_OFFLINE,
      use_fixed_point, num_frac_bits, N_TOH * N_TOW);
  LOG(INFO) << "Tiled weights size: " << tiled_weights.size();

#ifdef DUMP_ARRAY
  DumpArray<T>("input.txt", input.data(), input.size());
  DumpArray<T>("tiled_input.txt", tiled_input.data(), tiled_input.size());
  DumpArray<T>("weights.txt", weights.data(), weights.size());
  DumpArray<T>("tiled_weights.txt", tiled_weights.data(), tiled_weights.size());
#endif

  // this tiled output should be further reduced
  std::vector<T> tiled_output(N_T * output_tile_size);

  // create actions for DFE call
  typename DfeT::dfe_run_actions_t actions;
  actions.param_batch_size = N_T;

  if (DFE_COEFF_ON_CHIP)
    SplitCoeffAndAssign<T>((double **)(&(actions.param_batch_size) + 1),
                           weights.data(), cp);
#ifndef USE_DRAM
  actions.instream_ifmap = tiled_input.data();
  actions.instream_coeff_0 = tiled_weights.data();
  actions.outstream_ofmap = reinterpret_cast<T *>(tiled_output.data());
#endif

#ifdef USE_DRAM
  constexpr size_t num_bytes_per_burst = 384;
  size_t base_addr = 0;

  BurstAlign(tiled_input,
             num_bytes_per_burst * DFE_PC *
                 (DFE_USE_WINO ? WINO_TILE_SIZE * WINO_TILE_SIZE : DFE_PK));
  LOG(INFO) << "Tiled input size (burst aligned): " << tiled_input.size();
  BurstAlign(tiled_weights, num_bytes_per_burst * DFE_PC * DFE_PF);
  LOG(INFO) << "Tiled weights size (burst aligned): " << tiled_weights.size();
  BurstAlign(tiled_output, num_bytes_per_burst * DFE_PF *
                               (DFE_USE_WINO ? DFE_POH * DFE_POW : DFE_PK));

  base_addr = WriteDRAM<T, DfeT>(tiled_input, base_addr, engine);
  if (!DFE_COEFF_ON_CHIP)
    base_addr = WriteDRAM<T, DfeT>(tiled_weights, base_addr, engine);
#endif

  LOG(INFO) << "Running " << N_T << " convolution on DFE ...";
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  DfeT::Run(engine, &actions);
  end = std::chrono::system_clock::now();
  LOG(INFO) << "Done";

#ifdef USE_DRAM
  ReadDRAM<T, DfeT>(tiled_output, base_addr, engine);
#endif

  // transform the the final output
  output = TransformConvLayerTiledOutput<T>(tiled_output, OH, OW, F, T_OH, T_OW,
                                            TF, DFE_PF, DFE_POH, DFE_POW, N_TC,
                                            use_fixed_point, num_frac_bits);

  std::chrono::duration<double> elapsed_seconds = end - start;
  LOG(INFO) << "elapsed time: " << elapsed_seconds.count() << "s";
  auto gflops =
      (2.0 * OH * OW * C * F * K * K) / (elapsed_seconds.count()) * 1e-9;
  auto tiled_gflops = (2.0 * T_OH * T_OW * TC * TF * K * K) * N_T /
                      (elapsed_seconds.count()) * 1e-9;
  LOG(INFO) << "GFLOPS: " << gflops;
  LOG(INFO) << "Tiled GFLOPS: " << tiled_gflops;
}
#endif
#endif
