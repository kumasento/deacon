#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

int main(int argc, char *argv[]) {
  const int N = 1024;
  const int K = 4;

  uint32_t *data_inp = (uint32_t *) malloc(sizeof(uint32_t) * N);
  uint32_t *data_wgt = (uint32_t *) malloc(sizeof(uint32_t) * N);
  uint32_t *data_out = (uint32_t *) malloc(sizeof(uint32_t) * N / (K * K));

  for (int i = 0; i < N; i ++) {
    data_inp[i] = i + 1;
    data_wgt[i] = i + 1;
  }

  MaxDeep(N, data_inp, data_wgt, data_out);

  for (int i = 0; i < N / (K * K); i ++)
    printf("out[%4d] = %4d (%4d)\n", i, data_out[i], data_inp[i]);

  return 0;
}
