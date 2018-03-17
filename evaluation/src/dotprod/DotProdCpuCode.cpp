/**
 * Test code for the dot product design.
 * \author Ruizhe Zhao
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <glog/logging.h>
#include <getopt.h>

#include "Maxfiles.h"

max_file_t *max_file;
max_engine_t *max_engine;

template <typename T>
std::vector<T> dotprod_cpu(std::vector<T> vec_a, std::vector<T> vec_b,
                           int num_vecs) {
  auto num_vec_elems =
      static_cast<int>(static_cast<double>(vec_a.size()) / num_vecs);

  CHECK_EQ(vec_a.size(), vec_b.size()) << "input vectors length should match";

  std::vector<T> result(num_vecs);
  for (int i = 0; i < num_vecs; i++) {
    result[i] = static_cast<T>(0.0f);
    for (int j = 0; j < num_vec_elems; j++) {
      int idx = i * num_vec_elems + j;
      result[i] += vec_a[idx] * vec_b[idx];
    }
  }

  return result;
}

template <typename T>
std::vector<T> dotprod_dfe(std::vector<T> vec_a, std::vector<T> vec_b,
                           int num_vecs, const uint64_t VEC_LEN) {
  std::cout << "Starting DFE ..." << std::endl;
  auto num_vec_elems =
      static_cast<int>(static_cast<double>(vec_a.size()) / num_vecs);

  CHECK_EQ(vec_a.size(), vec_b.size()) << "input vectors length should match";

  std::vector<T> result(num_vecs);

  DotProd_actions_t actions;
  actions.instream_VEC_A = vec_a.data();
  actions.instream_VEC_B = vec_b.data();
  actions.outstream_RESULT = (T *)result.data();
  actions.param_N = num_vec_elems;
  actions.param_M = num_vecs;

  DotProd_run(max_engine, &actions);

  return result;
}

typedef float T;

int main(int argc, char *argv[]) {
  max_file = DotProd_init();
  max_engine = max_load(max_file, "*");

  const uint64_t VEC_LEN = max_get_constant_uint64t(max_file, "VEC_LEN");

  int num_vec_elems = 1024;
  int num_vecs = 16;
  int total_vec_elems = num_vec_elems * num_vecs;

  std::cout << "Total number of vectors: " << num_vecs << std::endl;
  std::cout << "Number of elements in each vector: " << num_vec_elems
            << std::endl;
  std::cout << "Total number of vector elements: " << total_vec_elems
            << std::endl;

  std::vector<T> vec_a(total_vec_elems), vec_b(total_vec_elems);
  for (int i = 0; i < total_vec_elems; i++) {
    vec_a[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    vec_b[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
  }

  // Run CPU
  auto start = std::chrono::system_clock::now();
  auto golden = dotprod_cpu<T>(vec_a, vec_b, num_vecs);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "CPU result: " << std::endl;
  for (int i = 0; i < (int)golden.size(); i++)
    printf("golden[%4d] = %.6f\n", i, golden[i]);
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;

  // Run DFE
  start = std::chrono::system_clock::now();
  auto result = dotprod_dfe<T>(vec_a, vec_b, num_vecs, VEC_LEN);
  end = std::chrono::system_clock::now();

  elapsed = end - start;
  std::cout << "DFE result: " << std::endl;
  for (int i = 0; i < (int)result.size(); i++)
    printf("result[%4d] = %.6f\n", i, result[i]);
  std::cout << "elapsed time: " << elapsed.count() << " sec" << std::endl;

  // Test
  CHECK_EQ(result.size(), golden.size());
  for (int i = 0; i < (int)result.size(); i++)
    CHECK_LT(fabs((golden[i] - result[i]) / golden[i]), 1e-6);

  max_unload(max_engine);

  return 0;
}
