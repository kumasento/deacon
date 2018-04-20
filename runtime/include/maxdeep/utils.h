/**
 * Utility functions for MaxDeep
 */

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

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

/*! Get number of tiles */
int GetNumTiles(int num_elems, int tile_size) {
  return static_cast<int>(ceil(static_cast<float>(num_elems) / tile_size));
}

template <typename T>
std::vector<T> CreateRandomArray(int N, int min_val = 0, int max_val = 1) {
  CHECK_GT(N, 0);
  CHECK_GE(max_val, min_val);

  std::vector<T> arr(N);
  auto range = max_val - min_val;

  for (int i = 0; i < N; i++) {
    float rand_val = static_cast<float>(rand()) / RAND_MAX;
    rand_val = rand_val * range + min_val;

    arr[i] = static_cast<T>(rand_val);
  }

  return arr;
}

template <typename T>
std::vector<T> CreateRandomTensor(int C, int H, int W, int pad_size = 0,
                                  int min_val = 0, int max_val = 1) {
  CHECK_GT(C, 0);
  CHECK_GT(H, 0);
  CHECK_GT(W, 0);
  CHECK_GE(pad_size, 0);

  std::vector<T> arr(C * H * W);
  auto range = max_val - min_val;

  for (int c = 0; c < C; c++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        auto i = c * H * W + h * W + w;

        if (h < pad_size || h >= H - pad_size || w < pad_size ||
            w >= W - pad_size)
          arr[i] = static_cast<T>(0.0f);
        else {
          float rand_val = static_cast<float>(rand()) / RAND_MAX;
          rand_val = rand_val * range + min_val;
          arr[i] = static_cast<T>(rand_val);
        }
      }
    }
  }

  return arr;
}
