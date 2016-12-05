#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

#define H 4
#define W 4

float input[H * W];
float output[H * W];

int main(int argc, char *argv[]) {

  for (int i = 0; i < H * W; i ++)
    input[i] = (float) i;

  Network(input, output);
  
  for (int i = 0; i < W; i ++)
    printf("Output[%6d] = %lf\n", i, output[i]);

  return 0;
}
