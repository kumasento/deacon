#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

#define H 4
#define W 4
#define F 4
#define C 2
#define K 3
#define M 8
#define N (H * W * F)

#define INPUT_SIZE (H * W * C)
#define OUTPUT_SIZE (H * W * F)
#define CONV_WGT_SIZE (F * K * K)
#define FC_OUT_SIZE (M)
#define FC_WGT_SIZE (M * N)

float input [INPUT_SIZE];
float output [FC_OUT_SIZE];
float conv_wgt [CONV_WGT_SIZE];
float fc_wgt [FC_WGT_SIZE];

int main(int argc, char *argv[]) {

  for (int i = 0; i < INPUT_SIZE; i ++) 
    input[i] = (float) 1;
  for (int i = 0; i < CONV_WGT_SIZE; i ++)
    conv_wgt[i] = (float) 1;
  for (int i = 0; i < FC_WGT_SIZE; i ++)
    fc_wgt[i] = (float) 1;

  Network(conv_wgt, fc_wgt, input, output);
  
  for (int i = 0; i < FC_OUT_SIZE; i ++) {
    if (i % (H * W) == 0)
      printf("\n");
    printf("Output[%6d] = %lf\n", i, output[i]);
  }

  return 0;
}
