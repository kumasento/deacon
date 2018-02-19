/**
 * Layers
 */

template <typename T>
void depthwise_separable_conv_layer(T *ifmap, T *coeff, T *ofmap, int H, int W,
                                    int C, int F, int K, int batch_size) {
  // TODO: remove this assertion
  if (batch_size != 1) exit(1);

  T *depth_coeff = coeff;
  T *point_coeff = &coeff[C * K * K];

  int OH = H - K + 1;
  int OW = W - K + 1;

  for (int f = 0; f < F; f++) {
    for (int oh = 0; oh < OH; oh++) {
      for (int ow = 0; ow < OW; ow++) {
        int out_idx = f * OH * OW + oh * OW + ow;
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
