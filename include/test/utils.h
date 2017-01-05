#ifndef TEST_UTILS_H__
#define TEST_UTILS_H__

#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <cstdio>
#include "glog/logging.h"

// The test data dir is a relative data path that assumes all the 
// test will be run in the build directory
#define TEST_DATA_DIR "../data/test"

template <typename T>
std::vector<T> read_test_data(std::string test_name, std::string data_name, bool aligned=false) {
  std::string file_name = std::string(TEST_DATA_DIR);
  file_name.append("/");
  file_name.append(test_name);
  file_name.append("/");
  file_name.append(data_name);
  file_name.append(".bin");

  LOG(INFO) << "Reading from file: " + file_name;

  std::vector<T> data;
  std::ifstream data_file(file_name, std::ios::binary | std::ios::ate);
  if (data_file.is_open()) {
    int size = data_file.tellg();
    int orig = size;
    
    if (size % 16 != 0 && aligned)
      size = (orig / 16 + 1) * 16;

    char buf[sizeof(T)];
    int N = size / sizeof(T);

    LOG(INFO) << "Number of bytes: " << size;
    LOG(INFO) << "Number of elements: " << N;

    // read
    data.resize(N);
    data_file.seekg(std::ios::beg);
    for (int i = 0; i < orig/sizeof(T); i ++) {
      T val;
      data_file.read(buf, sizeof(T));
      memcpy(&val, buf, sizeof(T));
      data[i] = val;
    }
  } else {
    LOG(FATAL) << file_name << " cannot be opened";
  }

  return data;
}

double * convert_to_double(std::vector<float> orig_data) {
  double *data = (double *) malloc(sizeof(double) * orig_data.size());
  for (int i = 0; i < orig_data.size(); i ++)
    data[i] = (double) orig_data[i];
  return data;
}

#endif
