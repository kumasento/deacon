#include <vector>
#include <string>
// google
#include "gtest/gtest.h"
#include "glog/logging.h"

// maxeler
#include "test_alexnet_MaxDeep.h"

// protobuf
#include "maxdeep.pb.h"

// self defined
#include "test/utils.h"

#define MAX_MEM_SIZE 31200
#define TEST_NAME "test_alexnet"

TEST(AlexNetTest, MainTest) {
  std::vector<float> inp_data   = read_test_data<float>(std::string(TEST_NAME), std::string("data"));
  std::vector<float> conv1_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_param_1"));
  std::vector<float> conv1_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_param_0"));
  std::vector<float> conv2_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_param_1"));
  std::vector<float> conv2_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_param_0"));
  std::vector<float> conv3_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_param_1"));
  std::vector<float> conv3_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_param_0"));
  std::vector<float> conv4_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_param_1"));
  std::vector<float> conv4_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_param_0"));
  std::vector<float> conv5_bias = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_param_1"));
  std::vector<float> conv5_wgts = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_param_0"));
  std::vector<float> fc6_bias   = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_1"));
  std::vector<float> fc6_wgts   = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_0"));
  std::vector<float> fc7_bias   = read_test_data<float>(std::string(TEST_NAME), std::string("fc7_param_1"));
  std::vector<float> fc7_wgts   = read_test_data<float>(std::string(TEST_NAME), std::string("fc7_param_0"));
  std::vector<float> fc8_bias   = read_test_data<float>(std::string(TEST_NAME), std::string("fc8_param_1"));
  std::vector<float> fc8_wgts   = read_test_data<float>(std::string(TEST_NAME), std::string("fc8_param_0"));
  std::vector<float> opt_data   = read_test_data<float>(std::string(TEST_NAME), std::string("fc8"));

  float *opt = (float *) malloc(sizeof(float) * opt_data.size());

  test_alexnet_MaxDeep_actions_t actions;
  actions.inmem_conv1_acc_bias = convert_to_double(conv1_bias);
  actions.inmem_conv2_acc_bias = convert_to_double(conv2_bias);
  actions.inmem_conv3_acc_bias = convert_to_double(conv3_bias);
  actions.inmem_conv4_acc_bias = convert_to_double(conv4_bias);
  actions.inmem_conv5_acc_bias = convert_to_double(conv5_bias);
  actions.instream_conv1_wgts  = conv1_wgts.data();
  actions.instream_conv2_wgts  = conv2_wgts.data();
  actions.instream_conv3_wgts  = conv3_wgts.data();
  actions.instream_conv4_wgts  = conv4_wgts.data();
  actions.instream_conv5_wgts  = conv5_wgts.data();
  actions.instream_fc6_bias    = fc6_bias.data();
  actions.instream_fc6_wgts    = fc6_wgts.data();
  actions.instream_fc7_bias    = fc7_bias.data();
  actions.instream_fc7_wgts    = fc7_wgts.data();
  actions.instream_fc8_bias    = fc8_bias.data();
  actions.instream_fc8_wgts    = fc8_wgts.data();
  actions.instream_cpu_inp     = inp_data.data();
  actions.outstream_cpu_out    = opt;

  max_file_t *max_file = test_alexnet_MaxDeep_init();
  max_engine_t *engine = max_load(max_file, "local:*");

  LOG(INFO) << "Running test";
  test_alexnet_MaxDeep_run(engine, &actions);
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
