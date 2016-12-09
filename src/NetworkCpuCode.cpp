#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

#define H 4
#define W 4
#define F 2
#define C 3

#define INPUT_SIZE (H * W * C)
#define OUTPUT_SIZE (H * W * F)

float input[INPUT_SIZE];
float output[OUTPUT_SIZE];

int main(int argc, char *argv[]) {

  for (int i = 0; i < INPUT_SIZE; i ++)
    input[i] = (float) (i);

  Network(input, output);
  
  for (int i = 0; i < OUTPUT_SIZE; i ++) {
    if (i % (H * W) == 0)
      printf("\n");
    printf("Output[%6d] = %lf\n", i, output[i]);
  }

  return 0;
}
