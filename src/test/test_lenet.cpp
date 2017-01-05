#include <vector>
#include <string>
// google
#include "gtest/gtest.h"
#include "glog/logging.h"

// maxeler
#include "test_lenet_MaxDeep.h"

// protobuf
#include "maxdeep.pb.h"

// self defined
#include "test/utils.h"

#define TEST_NAME "test_lenet"

TEST(LeNetTest, MainTest) {
  std::vector<float> inp_data  =
    read_test_data<float>(std::string(TEST_NAME), std::string("data"), true);
  std::vector<float> conv1_bias = 
    read_test_data<float>(std::string(TEST_NAME), std::string("conv1_param_1"));
  std::vector<float> conv1_wgt = 
    read_test_data<float>(std::string(TEST_NAME), std::string("conv1_param_0"));
  std::vector<float> conv2_bias = 
    read_test_data<float>(std::string(TEST_NAME), std::string("conv2_param_1"));
  std::vector<float> conv2_wgt = 
    read_test_data<float>(std::string(TEST_NAME), std::string("conv2_param_0"));
  std::vector<float> ip1_bias = 
    read_test_data<float>(std::string(TEST_NAME), std::string("ip1_param_1"));
  std::vector<float> ip1_wgt = 
    read_test_data<float>(std::string(TEST_NAME), std::string("ip1_param_0"));
  std::vector<float> ip2_bias = 
    read_test_data<float>(std::string(TEST_NAME), std::string("ip2_param_1"));
  std::vector<float> ip2_wgt = 
    read_test_data<float>(std::string(TEST_NAME), std::string("ip2_param_0"));
  std::vector<float> opt_data
    = read_test_data<float>(std::string(TEST_NAME), std::string("ip2"), true);

  float *opt = (float *) malloc(sizeof(float) * opt_data.size());

  test_lenet_MaxDeep_actions_t actions;
  actions.inmem_conv1_acc_bias = convert_to_double(conv1_bias);
  actions.inmem_conv2_acc_bias = convert_to_double(conv2_bias);
  actions.inmem_conv1_inp_wgts = convert_to_double(conv1_wgt);
  actions.inmem_conv2_inp_wgts = convert_to_double(conv2_wgt);
  actions.inmem_ip1_inp_bias   = convert_to_double(ip1_bias);
  actions.inmem_ip2_inp_bias   = convert_to_double(ip2_bias);
  actions.inmem_ip1_inp_wgts   = convert_to_double(ip1_wgt);
  actions.inmem_ip2_inp_wgts   = convert_to_double(ip2_wgt);
  actions.instream_cpu_inp     = inp_data.data();
  actions.outstream_cpu_out    = opt;

  max_file_t *max_file = test_lenet_MaxDeep_init();
  max_engine_t *engine = max_load(max_file, "local:*");

  LOG(INFO) << "Running test";
  test_lenet_MaxDeep_run(engine, &actions);
  LOG(INFO) << "Done!";

  for (int i = 0; i < opt_data.size(); i ++)
    ASSERT_NEAR(opt_data[i], opt[i], 1e-2);

  LOG(INFO) << "Passed";

  max_unload(engine);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
