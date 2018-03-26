template <typename T, int D0, int D1, int D2, int D3, int H, int W, int K>
void bottleneck(T in0[D0][H][W], T wght0[D1][D0], T wght1[D2][D1][K][K],
                T wght2[D3][D2]) {
  T in1[D1][H][W];
  T in2[D2][H][W];
  T in3[D3][H][W];

  // pointwise
  for (int d1 = 0; d1 < D1; d1++) {
    for (int d0 = 0; d0 < D0; d0++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          in1[d1][h][w] += in0[d0][h][w] * wght0[d1][d0];
        }
      }
    }
  }

  // standard convolution
  for (int d2 = 0; d2 < D2; d2++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int d1 = 0; d1 < D1; d1++) {
          for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
              int ih = h + kh - 1;
              int iw = w + kw - 1;

              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

              in2[d2][h][w] += in1[d1][ih][iw] * wght1[d2][d1][kh][kw];
            }
          }
        }
      }
    }
  }

  // pointwise
  for (int d3 = 0; d3 < D3; d3++) {
    for (int d2 = 0; d2 < D2; d2++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          in3[d3][h][w] += in2[d2][h][w] * wght2[d3][d2];
        }
      }
    }
  }
}
