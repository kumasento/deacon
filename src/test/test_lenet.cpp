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
    read_test_data<float>(std::string(TEST_NAME), std::string("data"));
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
    = read_test_data<float>(std::string(TEST_NAME), std::string("ip2"));

  float *opt = (float *) malloc(sizeof(float) * opt_data.size());

  test_lenet_MaxDeep_actions_t actions;
  actions.instream_conv1_bias = conv1_bias.data();
  actions.instream_conv1_wgt  = conv1_wgt.data();
  actions.instream_conv2_bias = conv2_bias.data();
  actions.instream_conv2_wgt  = conv2_wgt.data();
  actions.instream_ip1_bias   = ip1_bias.data();
  actions.instream_ip1_wgt    = ip1_wgt.data();
  actions.instream_ip2_bias   = ip2_bias.data();
  actions.instream_ip2_wgt    = ip2_wgt.data();
  actions.instream_cpu_inp    = inp_data.data();
  actions.outstream_cpu_out   = opt;

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
