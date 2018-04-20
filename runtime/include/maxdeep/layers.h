/**
 * Layers - implemented software version of various CNN layers.
 */

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <glog/logging.h>

#include "maxdeep/utils.h"

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
 */
template <typename T>
std::vector<T> CreateConvLayerTiledInput(std::vector<T> &input, int H, int W,
                                         int C, int F, int K, int P, int S,
                                         int TH, int TW, int TC, int TF,
                                         bool dfe = true) {
  std::vector<T> tiled_input;

  if (dfe) {
    // simply do padding here
    auto DFE_TH = GetConvLayerInputDim(TH, K, 0, S);
    auto DFE_TW = GetConvLayerInputDim(TW, K, 0, S);
    auto tiled_input_size = DFE_TH * DFE_TW * TC;
    tiled_input.resize(tiled_input_size);

    for (int c = 0; c < TC; c++) {
      for (int h = 0; h < DFE_TH; h++) {
        for (int w = 0; w < DFE_TW; w++) {
          auto i = c * DFE_TH * DFE_TW + h * DFE_TW + w;
          auto j = c * TH * TW + (h - P) * TW + (w - P);

          if (h < P || h >= DFE_TH - P || w < P || w >= DFE_TW - P)
            tiled_input[i] = static_cast<T>(0.0f);
          else
            tiled_input[i] = input[j];

          // printf("c = %3d h = %3d w = %3d i = %5d j = %5d res = %6d\n", c, h,
          // w, i, j, tiled_input[i]);
        }
      }
    }
  }

  return tiled_input;
}
