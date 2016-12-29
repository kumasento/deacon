#include "gtest/gtest.h"
#include "test_conv_layer_MaxDeep.h"
#include "maxdeep.pb.h"
#include "glog/logging.h"

TEST(ConvLayer, SimpleTest) {
  LOG(INFO) << "Test conv layer";
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
