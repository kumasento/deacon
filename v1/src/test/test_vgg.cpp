#include <vector>
#include <string>
// google
#include "gtest/gtest.h"
#include "glog/logging.h"

// maxeler
#include "test_vgg_MaxDeep.h"

// protobuf
#include "maxdeep.pb.h"

// self defined
#include "test/utils.h"

#define MAX_MEM_SIZE 31200
#define TEST_NAME "test_vgg"

TEST(VGGTest, MainTest) {
  LOG(INFO) << TEST_NAME;

  // File read {{{
  std::vector<float> inp_data          = read_test_data<float>(std::string(TEST_NAME), std::string("data"));
  std::vector<float> opt_data          = read_test_data<float>(std::string(TEST_NAME), std::string("fc8"));
  std::vector<float> fc6_wgts          = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_0"));
  std::vector<float> fc6_bias          = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_1"));
  std::vector<float> fc7_wgts          = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_0"));
  std::vector<float> fc7_bias          = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_1"));
  std::vector<float> fc8_wgts          = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_0"));
  std::vector<float> fc8_bias          = read_test_data<float>(std::string(TEST_NAME), std::string("fc6_param_1"));
  std::vector<float> conv1_1_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_1_param_0"));
  std::vector<float> conv1_1_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_1_param_1"));
  std::vector<float> conv1_2_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_2_param_0"));
  std::vector<float> conv1_2_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv1_2_param_1"));
  std::vector<float> conv2_1_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_1_param_0"));
  std::vector<float> conv2_1_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_1_param_1"));
  std::vector<float> conv2_2_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_2_param_0"));
  std::vector<float> conv2_2_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv2_2_param_1"));
  std::vector<float> conv3_1_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_1_param_0"));
  std::vector<float> conv3_1_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_1_param_1"));
  std::vector<float> conv3_2_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_2_param_0"));
  std::vector<float> conv3_2_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_2_param_1"));
  std::vector<float> conv3_3_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_3_param_0"));
  std::vector<float> conv3_3_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv3_3_param_1"));
  std::vector<float> conv4_1_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_1_param_0"));
  std::vector<float> conv4_1_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_1_param_1"));
  std::vector<float> conv4_2_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_2_param_0"));
  std::vector<float> conv4_2_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_2_param_1"));
  std::vector<float> conv4_3_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_3_param_0"));
  std::vector<float> conv4_3_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv4_3_param_1"));
  std::vector<float> conv5_1_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_1_param_0"));
  std::vector<float> conv5_1_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_1_param_1"));
  std::vector<float> conv5_2_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_2_param_0"));
  std::vector<float> conv5_2_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_2_param_1"));
  std::vector<float> conv5_3_wgts_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_3_param_0"));
  std::vector<float> conv5_3_bias_data = read_test_data<float>(std::string(TEST_NAME), std::string("conv5_3_param_1"));
  // }}}
  // Setup action {{{
  float *opt = (float *) malloc(sizeof(float) * opt_data.size());

  test_vgg_MaxDeep_actions_t actions;
  actions.instream_cpu_inp       = inp_data.data();
  actions.outstream_cpu_out      = opt;
  actions.instream_fc6_wgts      = fc6_wgts.data();
  actions.instream_fc6_bias      = fc6_bias.data();
  actions.instream_fc7_wgts      = fc7_wgts.data();
  actions.instream_fc7_bias      = fc7_bias.data();
  actions.instream_fc8_wgts      = fc8_wgts.data();
  actions.instream_fc8_bias      = fc8_bias.data();
  actions.inmem_conv1_1_inp_wgts = convert_to_double(conv1_1_wgts_data);
  actions.inmem_conv1_1_acc_bias = convert_to_double(conv1_1_bias_data);
  actions.inmem_conv1_2_inp_wgts = convert_to_double(conv1_2_wgts_data);
  actions.inmem_conv1_2_acc_bias = convert_to_double(conv1_2_bias_data);
  actions.inmem_conv2_1_inp_wgts = convert_to_double(conv2_1_wgts_data);
  actions.inmem_conv2_1_acc_bias = convert_to_double(conv2_1_bias_data);
  actions.inmem_conv2_2_inp_wgts = convert_to_double(conv2_2_wgts_data);
  actions.inmem_conv2_2_acc_bias = convert_to_double(conv2_2_bias_data);
  actions.inmem_conv3_1_inp_wgts = convert_to_double(conv3_1_wgts_data);
  actions.inmem_conv3_1_acc_bias = convert_to_double(conv3_1_bias_data);
  actions.inmem_conv3_2_inp_wgts = convert_to_double(conv3_2_wgts_data);
  actions.inmem_conv3_2_acc_bias = convert_to_double(conv3_2_bias_data);
  actions.inmem_conv3_3_inp_wgts = convert_to_double(conv3_3_wgts_data);
  actions.inmem_conv3_3_acc_bias = convert_to_double(conv3_3_bias_data);
  actions.inmem_conv4_1_inp_wgts = convert_to_double(conv4_1_wgts_data);
  actions.inmem_conv4_1_acc_bias = convert_to_double(conv4_1_bias_data);
  actions.inmem_conv4_2_inp_wgts = convert_to_double(conv4_2_wgts_data);
  actions.inmem_conv4_2_acc_bias = convert_to_double(conv4_2_bias_data);
  actions.inmem_conv4_3_inp_wgts = convert_to_double(conv4_3_wgts_data);
  actions.inmem_conv4_3_acc_bias = convert_to_double(conv4_3_bias_data);
  actions.inmem_conv5_1_inp_wgts = convert_to_double(conv5_1_wgts_data);
  actions.inmem_conv5_1_acc_bias = convert_to_double(conv5_1_bias_data);
  actions.inmem_conv5_2_inp_wgts = convert_to_double(conv5_2_wgts_data);
  actions.inmem_conv5_2_acc_bias = convert_to_double(conv5_2_bias_data);
  actions.inmem_conv5_3_inp_wgts = convert_to_double(conv5_3_wgts_data);
  actions.inmem_conv5_3_acc_bias = convert_to_double(conv5_3_bias_data);
  // }}}
  // Setup Max Engine {{{
  max_file_t *max_file = test_vgg_MaxDeep_init();
  max_engine_t *engine = max_load(max_file, "local:*");

  LOG(INFO) << "Running test";
  test_vgg_MaxDeep_run(engine, &actions);
  LOG(INFO) << "Done!";

  for (int i = 0; i < opt_data.size(); i ++)
    ASSERT_NEAR(opt_data[i], opt[i], 1e-2);

  LOG(INFO) << "Passed";

  max_unload(engine);
  // }}}
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
