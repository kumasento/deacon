#include <iostream>

#include "Maxfiles.h"

int main(int argc, char *argv[]) {
  std::cout << "Testing MatMul Op in MaxDeep ..." << std::endl;

  max_file_t *max_file = MatMul_init();
  max_engine_t* engine = max_load(max_file, "*");

  const size_t H = 32, W = 32, N = 1;
  const size_t x_num_elems = W * N;
  const size_t W_num_elems = H * W * N;
  const size_t y_num_elems = H * N;

  typedef uint8_t ptr_t;
  typedef int32_t dat_t;

  ptr_t *x = (ptr_t*) malloc(sizeof(dat_t) * x_num_elems);
  ptr_t *w = (ptr_t*) malloc(sizeof(dat_t) * W_num_elems);
  ptr_t *y = (ptr_t*) malloc(sizeof(dat_t) * y_num_elems);

  MatMul_actions_t actions;
  actions.param_N = N;
  actions.instream_x = (const ptr_t *) x;
  actions.instream_W = (const ptr_t *) w;
  actions.outstream_y = y;

  std::cout << "Start to run" << std::endl;
  MatMul_run(engine, &actions);
  std::cout << "End" << std::endl;

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
