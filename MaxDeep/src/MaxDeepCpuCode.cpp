#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

int main(int argc, char *argv[]) {
  const int K = 2;
  const int N = 2;
  const int data_size = 384 * K * K * N;

  uint8_t *data_inp = (uint8_t *) malloc(sizeof(uint32_t) * data_size);
  uint8_t *data_wgt = (uint8_t *) malloc(sizeof(uint32_t) * data_size);
  uint8_t *data_out = (uint8_t *) malloc(sizeof(uint32_t) * data_size / (K * K));

  // TODO: write stream encoder and decoder methods to transform the
  // data type
  // (2017-02-24) Ruizhe Zhao
  for (int i = 0; i < data_size * 4; i += 4) {
    uint32_t val = i / 4 + 1;
    for (int j = 0; j < 4; j ++) {
      data_inp[i + j] = (val >> (j * 8));
      data_wgt[i + j] = (val >> (j * 8));
    }
  }

  MaxDeep_dramWrite(data_size * sizeof(uint32_t), 0, data_inp);
  MaxDeep_dramWrite(data_size * sizeof(uint32_t), data_size * sizeof(uint32_t), data_wgt);
  MaxDeep(data_size);
  MaxDeep_dramRead(data_size / (K * K) * sizeof(uint32_t), data_size * 2 * sizeof(uint32_t), data_out);

  for (int i = 0; i < 16; i += 4) {
    uint32_t val = 0;
    for (int j = 0; j < 4; j ++)
      val += (data_out[i + j] << (j * 8));
    printf("out[%4d] = %u\n", i / 4, val);
  }

  return 0;
}
