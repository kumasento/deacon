/*! This is an example CPU code.
 */

#include <iostream>
#include <cstdlib>
#include <cmath>

#include "Maxfiles.h"

typedef float T;

int main(int argc, char *argv[]) {
  max_file_t *max_file = Max5StrmMult_init();
  max_engine_t *engine = max_load(max_file, "*");

  const int N = 1024;
  T *x, *y, *z;

  x = (T *)malloc(sizeof(T) * N);
  y = (T *)malloc(sizeof(T) * N);
  z = (T *)malloc(sizeof(T) * N);

  for (int i = 0; i < N; i++) {
    x[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    y[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
  }

  Max5StrmMult_actions_t actions;
  // should be revised
  actions.param_N = N;
  actions.instream_x = (const T *)x;
  actions.instream_y = (const T *)y;
  actions.outstream_z = z;

  Max5StrmMult_run(engine, &actions);

  for (int i = 0; i < N; i++) {
    printf("[%04d] %10.6f * %10.6f = %10.6f\n", i, x[i], y[i], z[i]);
    if (std::abs(x[i] * y[i] - z[i]) > 1e-6) {
      fprintf(stderr, "ERROR at %4d: golden %10.6f result %10.6f\n", i,
              x[i] * y[i], z[i]);
      exit(1);
    }
  }

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
