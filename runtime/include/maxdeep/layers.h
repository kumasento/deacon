/**
 * Layers
 */

#include <math.h>
#include <stdlib.h>
#include <algorithm>

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

template <typename T, int T_H, int T_W, int T_C, int T_F>
void conv_layer(T *input, T *weights, T *bias, T *output, int H, int W, int C,
                int F, int K, int P, int S) {
  auto O_H = (H - K + 2 * P) / S + 1;
  auto O_W = (W - K + 2 * P) / S + 1;

  auto num_tile_h = static_cast<int>(ceil(static_cast<float>(O_H) / T_H));
  auto num_tile_w = static_cast<int>(ceil(static_cast<float>(O_W) / T_W));
  auto num_tile_c = static_cast<int>(ceil(static_cast<float>(C) / T_C));
  auto num_tile_f = static_cast<int>(ceil(static_cast<float>(F) / T_F));

  auto T_H_i = T_H * S + 2 * P;
  auto T_W_i = T_W * S + 2 * P;

  auto input_tile =
      reinterpret_cast<T *>(malloc(sizeof(T) * T_H_i * T_W_i * T_C));
  auto weight_tile =
      reinterpret_cast<T *>(malloc(sizeof(T) * T_F * T_C * K * K));
  auto bias_tile = reinterpret_cast<T *>(malloc(sizeof(T) * T_F));
  auto output_tile = reinterpret_cast<T *>(malloc(sizeof(T) * T_H * T_W * T_F));

  for (int t_f = 0; t_f < num_tile_f; t_f++) {
    for (int t_h = 0; t_h < num_tile_h; t_h++) {
      for (int t_w = 0; t_w < num_tile_w; t_w++) {
        auto num_f = std::min(T_F, F - t_f * T_F);
        auto num_h = std::min(T_H, H - t_h * T_H);
        auto num_w = std::min(T_W, W - t_w * T_W);

        for (int f_i = 0; f_i < num_f; f_i++) {
          for (int o_h_i = 0; o_h_i < num_h; o_h_i++) {
            for (int o_w_i = 0; o_w_i < num_w; o_w_i++) {
              auto f = f_i + t_f * T_F;
              auto o_h = o_h_i + t_h * T_H;
              auto o_w = o_w_i + t_w * T_W;
              auto o_i = f * O_H * O_W + o_h * O_W + o_w;
              auto o_i_i = f_i * T_H * T_W + o_h_i * T_W + o_w_i;

              output_tile[o_i_i] = bias_tile[f];

              for (int t_c = 0; t_c < num_tile_c; t_c++) {
                auto num_c = std::min(T_C, C - t_c * T_C);

                for (int c_i = 0; c_i < num_c; c_i++) {
                  auto c = c_i + t_c * T_C;
                  for (int k_h = 0; k_h < K; k_h++) {
                    for (int k_w = 0; k_w < K; k_w++) {
                      auto i_h_i = o_h_i * S + k_h - P;
                      auto i_w_i = o_w_i * S + k_w - P;

                      if (i_h_i < 0 || i_h_i >= T_H_i || i_w_i < 0 ||
                          i_w_i >= T_W_i)
                        continue;

                      auto i_i = c_i * T_H_i * T_W_i + i_h_i * T_W_i + i_w_i;
                      auto w_i =
                          f_i * T_C * K * K + c_i * K * K + k_h * K + k_w;
                      output_tile[o_i_i] += input_tile[i_i] * weight_tile[w_i];
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
}
