#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <time.h>
#include <getopt.h>

#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

#include "maxdeep/utils.h"
#include "maxdeep/test_runners.h"

int main(int argc, char *argv[]) {
  std::cout << "\x1B[33mMaxDeep Command Line Program\x1B[0m" << std::endl;
  std::cout << "Number of CLI args: " << argc << std::endl;
  for (int i = 0; i < argc; i ++)
    printf("ARGV[%3d] = %s\n", i, argv[i]);
  printf("\n");

  int c;
  int num_iters = 1;
  std::string design_name;
  while ((c = getopt(argc, argv, "n:i:")) != -1)
    switch (c) {
      case 'n':
        design_name = std::string(optarg);
        break;
      case 'i':
        num_iters = atoi(optarg);
        break;
      default:
        fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        exit(1);
    }

  printf("DESIGN_NAME: \x1B[32m%s\x1B[0m\n", design_name.c_str());

  bool is_sim = maxdeep::utils::is_sim(std::string(argv[0])); 
  if (is_sim)
    std::cout << "ENVIRONMENT: \x1B[32mSIMULATION\x1B[0m" << std::endl;

  printf("\x1B[32mLOADING\x1B[0m maxfile ...\n");
  max_file_t *maxfile = MaxDeep_init();
  max_engine_t *engine = max_load(maxfile, "*:0");

  if (design_name == std::string("LOOPBACK") ||
      design_name == std::string("LOOPBACK_PADDED"))
    maxdeep::test_runners::run_loopback_test(is_sim, maxfile, engine);
  else if (design_name == std::string("MULT_ARRAY"))
    maxdeep::test_runners::run_mult_array_test(is_sim, maxfile, engine);
  else if (design_name == std::string("ONE_DIM_CONV"))
    maxdeep::test_runners::run_one_dim_conv_test(is_sim, maxfile, engine);
  else if (design_name == std::string("CONV2D")) {
    int bitwidth = (int) max_get_constant_uint64t(maxfile, "BITWIDTH");
    printf("bitwidth: %d\n", bitwidth);
    if (bitwidth == 32) {
      maxdeep::test_runners::Conv2DTest<uint32_t> test(is_sim, maxfile, engine);
      test.run(num_iters);
    } else if (bitwidth == 16) {
      maxdeep::test_runners::Conv2DTest<uint16_t> test(is_sim, maxfile, engine);
      test.run(num_iters);
    }
  } else
    throw std::runtime_error("design_name cannot be recognised!");

  max_file_free(maxfile);
  max_unload(engine);

  return 0;
}

