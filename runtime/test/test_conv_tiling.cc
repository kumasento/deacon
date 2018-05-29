/*! Test the tiled arrays.
 */

#define NO_DFE
#include "maxdeep/layers.h"

#include <gmock/gmock.h>
#include <vector>

using ::testing::ElementsAreArray;

TEST(TestConvTiling, TestSingleTiledWeights) {
  std::vector<float> weights = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                13.0f, 14.0f, 15.0f, 16.0f};

  int C = 2, K = 2, PC = 2, F = 2, PF = 2;

  std::vector<float> tiled_weights =
      CreateConvLayerTiledWeights(weights, C, F, K, C, F, PC, PF);

  // Nothing should have changed
  ASSERT_THAT(tiled_weights, ElementsAreArray(weights));
}

TEST(TestConvTiling, TestSingleTiledInput) {
  // 2 x 2 x 2 -> 2 x 4 x 4
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> golden = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 1.0f, 5.0f, 2.0f, 6.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 3.0f, 7.0f, 4.0f, 8.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  int C = 2, H = 2, W = 2, PC = 2, K = 3;

  auto tiled_input =
      CreateConvLayerTiledInput(input, H, W, C, K, 1, 1, H, W, C, PC);

  ASSERT_THAT(tiled_input, ElementsAreArray(golden));
}

TEST(TestConvTiling, TestMultiTiledInput) {
  // check whether multiple tiles could work

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> golden = {// tile 1
                               0, 0, 0, 0, 1, 2, 0, 3, 4,
                               // tile 2
                               0, 0, 0, 1, 2, 0, 3, 4, 0,
                               // tile 3
                               0, 1, 2, 0, 3, 4, 0, 0, 0,
                               // tile 4
                               1, 2, 0, 3, 4, 0, 0, 0, 0};

  int H = 2, W = 2, C = 1, K = 3, P = 1, S = 1, T_OH = 1, T_OW = 1, PC = 1,
      TC = 1;
  std::vector<float> output = CreateConvLayerTiledInput<float>(
      input, H, W, C, K, P, S, T_OH, T_OW, TC, PC);

  ASSERT_THAT(output, ElementsAreArray(golden));
}
