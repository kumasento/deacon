/**
 * Utility functions for MaxDeep
 */
#ifndef MAXDEEP_UTILS_H
#define MAXDEEP_UTILS_H

#include <glog/logging.h>
#include <math.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"

int GetConvLayerInputDim(int output_dim, int K, int P, int S) {
  // sanity checks
  CHECK_GT(output_dim, 0);
  CHECK_GT(K, 0);
  CHECK_GE(P, 0);
  CHECK_GT(S, 0);

  return (output_dim - 1) * S + K - 2 * P;
}

int GetConvLayerOutputDim(int input_dim, int K, int P, int S) {
  // sanity checks
  CHECK_GT(input_dim, 0);
  CHECK_GT(K, 0);
  CHECK_GE(P, 0);
  CHECK_GT(S, 0);
  CHECK_EQ((input_dim - K + 2 * P) % S, 0);

  return (input_dim - K + 2 * P) / S + 1;
}

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

/*! Get number of tiles */
int GetNumTiles(int num_elems, int tile_size) {
  return static_cast<int>(ceil(static_cast<float>(num_elems) / tile_size));
}

template <typename T>
std::vector<T> CreateConstantArray(int N, T value) {
  CHECK_GT(N, 0);

  std::vector<T> arr(N);
  for (int i = 0; i < N; i++) arr[i] = static_cast<T>(value);
  return arr;
}

template <typename T>
std::vector<T> CreateRandomArray(int N, float min_val = 0, float max_val = 1) {
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
inline T RoundToNearest(float data) {
  return static_cast<T>(std::round(data));
}

template <typename T>
T FloatToFixed(float data, int num_frac_bits) {
  return static_cast<T>(
      RoundToNearest<T>(data * static_cast<float>(1 << num_frac_bits)));
}

template <typename T>
float FixedToFloat(T data, int num_frac_bits) {
  return static_cast<float>(data) / static_cast<float>(1 << num_frac_bits);
}

template <typename T>
std::vector<T> FloatToFixed(std::vector<float>& data, int num_frac_bits) {
  std::vector<T> arr(data.size());

  for (int i = 0; i < (int)data.size(); i++)
    arr[i] = FloatToFixed<T>(data[i], num_frac_bits);

  return arr;
}

template <typename T>
std::vector<float> FixedToFloat(std::vector<T>& data, int num_frac_bits) {
  std::vector<float> arr(data.size());

  for (int i = 0; i < (int)data.size(); i++)
    arr[i] = FixedToFloat(data[i], num_frac_bits);

  return arr;
}

template <typename T>
inline float ConvertToFloat(T data, bool use_fixed_point, int num_frac_bits) {
  return !use_fixed_point ? static_cast<float>(data)
                          : FixedToFloat<float>(data, num_frac_bits);
}

template <typename T>
inline T Saturate(int32_t num) {
  if (num > static_cast<int32_t>(std::numeric_limits<T>::max()))
    return std::numeric_limits<T>::max();
  if (num < static_cast<int32_t>(std::numeric_limits<T>::min()))
    return std::numeric_limits<T>::min();
  return static_cast<T>(num);
}

template <typename T>
inline T FixedPointAdd(T a, T b) {
  auto ans = static_cast<int32_t>(a) + static_cast<int32_t>(b);
  return Saturate<T>(ans);
}

template <typename T>
inline T FixedPointMul(T a, T b, int num_frac_bits) {
  int32_t round_value = 1 << (num_frac_bits - 1);
  auto tmp = static_cast<int32_t>(a) * static_cast<int32_t>(b);
  tmp += round_value;
  return Saturate<T>(tmp >> num_frac_bits);
}

template <typename T>
inline std::vector<T> FixedPointMatMul(std::vector<T>& A, std::vector<T>& B,
                                       int M, int N, int K, int num_frac_bits) {
  std::vector<T> C(M * N);

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < K; k++) {
        C[i * N + j] = FixedPointAdd<T>(
            C[i * N + j],
            FixedPointMul<T>(A[i * K + k], B[k * N + j], num_frac_bits));
      }

  return C;
}

template <typename T>
inline std::vector<T> Transpose(std::vector<T>& A, int M, int N) {
  std::vector<T> B(M * N);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) B[j * M + i] = A[i * N + j];
  return B;
}

template <typename T>
inline void PrintMatrix(std::vector<T>& A, int M, int N, const char* name,
                        int num_frac_bits) {
  printf("%s = {\n", name);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%10.6f, ", FixedToFloat<T>(A[i * N + j], num_frac_bits));
    }
    printf("\n");
  }
  printf("}\n");
}

template <typename T>
inline std::vector<T> WinogradInputTransform(std::vector<T>& input, int R,
                                             bool use_fixed_point = false,
                                             int num_frac_bits = 0) {
  std::vector<T> trans(R * R);
  std::vector<float> B = {4,  0,  0,  0,  0,  0,   //
                          0,  -4, 4,  -2, 2,  4,   //
                          -5, -4, -4, -1, -1, 0,   //
                          0,  1,  -1, 2,  -2, -5,  //
                          1,  1,  1,  1,  1,  0,   //
                          0,  0,  0,  0,  0,  1};

  if (use_fixed_point) {
    std::vector<T> BB = FloatToFixed<T>(B, num_frac_bits);
    std::vector<T> BT = Transpose<T>(BB, R, R);
    std::vector<T> BTd = FixedPointMatMul<T>(BT, input, R, R, R, num_frac_bits);
    std::vector<T> BTdB = FixedPointMatMul<T>(BTd, BB, R, R, R, num_frac_bits);

#ifdef TRACE
    PrintMatrix<T>(BTdB, R, R, "BTdB", num_frac_bits);
    PrintMatrix<T>(BB, R, R, "BB", num_frac_bits);
    PrintMatrix<T>(BT, R, R, "BT", num_frac_bits);
    PrintMatrix<T>(input, R, R, "d", num_frac_bits);
    PrintMatrix<T>(BTd, R, R, "BTd", num_frac_bits);
#endif

    for (int i = 0; i < R * R; i++) trans[i] = BTdB[i];
  } else {
    LOG(FATAL) << "Only support fixed-point winograd offline computation";
  }

  return trans;
}

template <typename T>
inline std::vector<T> WinogradWeightsTransform(std::vector<T>& weights, int K,
                                               int R,
                                               bool use_fixed_point = false,
                                               int num_frac_bits = 0) {
  std::vector<T> trans(R * R);

  std::vector<float> G = {0.25f,    0.0f,     0.0f,     -1.f / 6,  -1.f / 6,
                          -1.f / 6, -1.f / 6, 1.f / 6,  -1.f / 6,  1.f / 24,
                          1.f / 12, 1.f / 6,  1.f / 24, -1.f / 12, 1.f / 6,
                          0.0f,     0.0f,     1.0f};

  if (use_fixed_point) {
    std::vector<T> GG = FloatToFixed<T>(G, num_frac_bits);
    std::vector<T> Gg =
        FixedPointMatMul<T>(GG, weights, R, K, K, num_frac_bits);
    std::vector<T> GT = Transpose<T>(GG, R, K);
    std::vector<T> GgG = FixedPointMatMul<T>(Gg, GT, R, R, K, num_frac_bits);

#ifdef TRACE
    PrintMatrix<T>(GG, R, K, "G", num_frac_bits);
    PrintMatrix<T>(weights, K, K, "g", num_frac_bits);
    PrintMatrix<T>(Gg, R, K, "Gg", num_frac_bits);
    PrintMatrix<T>(GT, K, R, "GT", num_frac_bits);
    PrintMatrix<T>(GgG, R, R, "GgG", num_frac_bits);
#endif

    for (int i = 0; i < R * R; i++) trans[i] = GgG[i];
  } else {
    LOG(FATAL) << "Only support fixed-point winograd offline computation";
  }

  return trans;
}

template <typename T>
inline std::vector<T> WinogradOutputTransform(std::vector<T>& output, int R,
                                              int M,
                                              bool use_fixed_point = false,
                                              int num_frac_bits = 0) {
  std::vector<T> trans(M * M);
  std::vector<float> A = {1, 0,  0, 0,   //
                          1, 1,  1, 1,   //
                          1, -1, 1, -1,  //
                          1, 2,  4, 8,   //
                          1, -2, 4, -8,  //
                          0, 0,  0, 1};

  if (use_fixed_point) {
    std::vector<T> AA = FloatToFixed<T>(A, num_frac_bits);
    std::vector<T> AT = Transpose<T>(AA, R, M);
    std::vector<T> ATo =
        FixedPointMatMul<T>(AT, output, M, R, R, num_frac_bits);
    std::vector<T> AToA = FixedPointMatMul<T>(ATo, AA, M, M, R, num_frac_bits);

#ifdef TRACE
    PrintMatrix<T>(AA, R, M, "A", num_frac_bits);
    PrintMatrix<T>(AT, M, R, "AT", num_frac_bits);
    PrintMatrix<T>(output, R, R, "o", num_frac_bits);
    PrintMatrix<T>(ATo, M, R, "ATo", num_frac_bits);
    PrintMatrix<T>(AToA, M, M, "AToA", num_frac_bits);
#endif

    for (int i = 0; i < M * M; i++) trans[i] = AToA[i];
  } else {
    LOG(FATAL) << "Only support fixed-point winograd offline computation";
  }

  return trans;
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

size_t GetBurstAlignedNumElems(size_t num_elems, size_t num_bytes_per_elem,
                               size_t num_bytes_per_burst) {
  CHECK(num_bytes_per_burst % num_bytes_per_elem == 0);

  auto total_num_bytes = num_elems * num_bytes_per_elem;
  auto num_bursts = static_cast<size_t>(
      std::ceil(static_cast<double>(total_num_bytes) / num_bytes_per_burst));
  auto burst_aligned_total_num_bytes = num_bursts * num_bytes_per_burst;

  return static_cast<size_t>(burst_aligned_total_num_bytes /
                             num_bytes_per_elem);
}

template <typename T>
void BurstAlign(std::vector<T>& arr, size_t num_bytes_per_burst) {
  auto burst_aligned_num_elems =
      GetBurstAlignedNumElems(arr.size(), sizeof(T), num_bytes_per_burst);
  LOG(INFO) << "Burst aligning input array of size: " << arr.size() << " to "
            << burst_aligned_num_elems << '\n';
  // the input vector is resized (side effect)
  arr.resize(burst_aligned_num_elems);
}

template <typename T>
void DumpArray(const char* file_name, T* data, int num, int num_frac_bits = 0) {
  std::ofstream out(file_name);

  if (!out) {
    fprintf(stderr, "Cannot open file for writing: %s\n", file_name);
    exit(1);
  }

  for (int i = 0; i < num; i++)
    out << (std::is_same<T, float>::value
                ? static_cast<float>(data[i])
                : FixedToFloat<T>(data[i], num_frac_bits))
        << std::endl;
}

#endif
