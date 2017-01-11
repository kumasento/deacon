#include <vector>
#include <string>
#include <chrono>
#include <ctime>
// google
#include "gtest/gtest.h"
#include "glog/logging.h"

// maxeler
#include "test_cifar_10_MaxDeep.h"

// protobuf
#include "maxdeep.pb.h"

// self defined
#include "test/utils.h"

#define MAX_MEM_SIZE 31200
#define TEST_NAME "test_cifar_10"

TEST(Cifar10Test, MainTest) {
  std::vector<float> inp_data   = read_test_data<float>(std::string(TEST_NAME), std::string("data"));
  std::vector<float> conv1_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_param_1"));
  std::vector<float> conv1_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_param_0"));
  std::vector<float> conv2_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_param_1"));
  std::vector<float> conv2_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_param_0"));
  std::vector<float> conv3_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_param_1"));
  std::vector<float> conv3_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_param_0"));
  std::vector<float> ip1_bias   = read_test_data<float>(std::string(TEST_NAME), std::string("ip1_param_1"));
  std::vector<float> ip1_wgts   = read_test_data<float>(std::string(TEST_NAME), std::string("ip1_param_0"));
  std::vector<float> opt_data   = read_test_data<float>(std::string(TEST_NAME), std::string("ip1"));

  float *opt = (float *) malloc(sizeof(float) * opt_data.size());

  test_cifar_10_MaxDeep_actions_t actions;
  actions.inmem_conv1_acc_bias = convert_to_double(conv1_bias);
  actions.inmem_conv2_acc_bias = convert_to_double(conv2_bias);
  actions.inmem_conv3_acc_bias = convert_to_double(conv3_bias);
  actions.instream_conv1_wgts  = conv1_wgts.data();
  actions.instream_conv2_wgts  = conv2_wgts.data();
  actions.instream_conv3_wgts  = conv3_wgts.data();
  actions.instream_ip1_bias    = ip1_bias.data();
  actions.instream_ip1_wgts    = ip1_wgts.data();
  actions.instream_cpu_inp     = inp_data.data();
  actions.outstream_cpu_out    = opt;

  max_file_t *max_file = test_cifar_10_MaxDeep_init();
  max_engine_t *engine = max_load(max_file, "local:*");

  LOG(INFO) << "Running test";
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  test_cifar_10_MaxDeep_run(engine, &actions);
  end = std::chrono::system_clock::now();
  LOG(INFO) << "Done!";

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  LOG(INFO) << "finished computation at " << std::ctime(&end_time)
    << "elapsed time: " << elapsed_seconds.count() << "s\n";

  for (int i = 0; i < opt_data.size(); i ++)
    ASSERT_NEAR(opt_data[i], opt[i], 1e-2);

  LOG(INFO) << "Passed";

  max_unload(engine);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}