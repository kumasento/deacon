/*! This is an example CPU code.
 */
#include <iostream>

#include "Maxfiles.h"

typedef uint8_t T;

int main(int argc, char *argv[]) {
  max_file_t *max_file = ShiftMultCompare_init();
  max_engine_t* engine = max_load(max_file, "*");

  const int N = 1024;
  T *x, *y, *z;

  x = (T *) malloc(sizeof(T) * N);
  y = (T *) malloc(sizeof(T) * N);
  z = (T *) malloc(sizeof(T) * N);

  ShiftMultCompare_actions_t actions;
  // should be revised
  actions.param_N = N;
  actions.instream_x = (const T *) x;
  actions.instream_y = (const T *) y;
  actions.outstream_z = z;

  ShiftMultCompare_run(engine, &actions);

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
