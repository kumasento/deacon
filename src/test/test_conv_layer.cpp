#include <vector>
#include <string>
// google
#include "gtest/gtest.h"
#include "glog/logging.h"

// maxeler
#include "test_conv_layer_MaxDeep.h"

// protobuf
#include "maxdeep.pb.h"

// self defined
#include "test/utils.h"

#define TEST_NAME "test_conv_layer"

TEST(SingleConvLayer, MainTest) {
  LOG(INFO) << "Test conv layer";

  std::vector<float> inp_data  = read_test_data<float>(std::string(TEST_NAME), std::string("data"));
  std::vector<float> opt_data  = read_test_data<float>(std::string(TEST_NAME), std::string("conv"));
  std::vector<float> wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv_param_0"));
  std::vector<float> bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv_param_1"));

  float *opt = (float *) malloc(sizeof(float) * opt_data.size());

  test_conv_layer_MaxDeep_actions_t actions;
  actions.instream_conv_wgt = wgts_data.data();
  actions.instream_cpu_inp = inp_data.data();
  actions.instream_conv_bias = bias_data.data();
  actions.outstream_cpu_out = opt;

  max_file_t *max_file = test_conv_layer_MaxDeep_init();
  max_engine_t *engine = max_load(max_file, "local:*");

  LOG(INFO) << "Running test";
  test_conv_layer_MaxDeep_run(engine, &actions);
  LOG(INFO) << "Done!";

  for (int i = 0; i < opt_data.size(); i ++)
    ASSERT_FLOAT_EQ(opt_data[i], opt[i]);
  LOG(INFO) << "Passed";

  max_unload(engine);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
