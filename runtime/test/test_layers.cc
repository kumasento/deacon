#include <stdlib.h>
#include "maxdeep/layers.h"

typedef float T;

int main(int argc, char *argv[]) {

  const int T_H = 16, T_W = 16, T_C = 8, T_F = 8;
  int H = 32, W = 32, C = 16, F = 32, K = 3, P = 1, S = 1;
  auto O_H = (H - K + 2 * P) / S + 1;
  auto O_W = (W - K + 2 * P) / S + 1;

  auto input = reinterpret_cast<T *>(malloc(sizeof(T) * H * W * C));
  auto weight = reinterpret_cast<T *>(malloc(sizeof(T) * K * K * C * F));
  auto bias = reinterpret_cast<T *>(malloc(sizeof(T) * F));
  auto output = reinterpret_cast<T *>(malloc(sizeof(T) * O_H * O_W * F));

  conv_layer<T, T_H, T_W, T_C, T_F>(input, weight, bias, output, H, W, C, F, K,
                                    P, S);

  free(input);
  free(weight);
  free(bias);
  free(output);

  return 0;
}
