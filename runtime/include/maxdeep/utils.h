/**
 * Utility functions for MaxDeep
 */

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>

template <typename T, int burst_size = 16>
T* create_array(int num, int* burst_aligned_num = nullptr) {
  int num_burst =
      static_cast<int>(ceil(static_cast<float>(num) * sizeof(T) / burst_size));
  int burst_aligned_size = num_burst * burst_size;
  if (burst_aligned_num != nullptr)
    *burst_aligned_num = burst_aligned_size / sizeof(T);

  auto arr = reinterpret_cast<T*>(malloc(burst_aligned_size));

  return arr;
}

template <typename T, int burst_size = 16>
T* random_initialize(int num, float scale = 1.0) {
  int burst_aligned_num;
  auto arr = create_array<T, burst_size>(num, &burst_aligned_num);

  for (int i = 0; i < burst_aligned_num; i++) {
    arr[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX * scale);
  }

  return arr;
}

template <typename T>
void dump_array(const char* file_name, T* data, int num) {
  std::ofstream out(file_name);

  if (!out) {
    fprintf(stderr, "Cannot open file for writing: %s\n", file_name);
    exit(1);
  }

  for (int i = 0; i < num; i++) out << static_cast<float>(data[i]) << std::endl;
}
