/**
 * Evaluation of Depthwise Separable Convolution.
 */

#include <getopt.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "Maxfiles.h"
#include "maxdeep/layers.h"
#include "maxdeep/utils.h"

#if BIT_WIDTH == 8
typedef int8_t data_t;
#elif BIT_WIDTH == 16
typedef int16_t data_t;
#elif BIT_WIDTH == 32
typedef int32_t data_t;
#endif

struct Dfe {
  typedef DepthwiseSeparableConvLayer_actions_t dfe_run_actions_t;
  typedef DepthwiseSeparableConvLayer_dramRead_actions_t dram_read_actions_t;
  typedef DepthwiseSeparableConvLayer_dramWrite_actions_t dram_write_actions_t;

  static void ReadDRAM(max_engine_t *engine, dram_read_actions_t *actions) {
    DepthwiseSeparableConvLayer_dramRead_run(engine, actions);
  }
  static void WriteDRAM(max_engine_t *engine, dram_write_actions_t *actions) {
    DepthwiseSeparableConvLayer_dramWrite_run(engine, actions);
  }
};

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  srand(42);

  char c;

  uint64_t batch_size = 1;
  uint64_t num_iters = 1;
  std::string data_file;
  while ((c = getopt(argc, argv, "i:n:f:")) != -1) {
    switch (c) {
      case 'i':
        num_iters = atoi(optarg);
        break;
      case 'n':
        batch_size = atoi(optarg);
        break;
      case 'f':
        data_file = std::string(optarg);
        break;
      default:
        exit(1);
    }
  }

  max_file_t *max_file = DepthwiseSeparableConvLayer_init();
  max_engine_t *engine = max_load(max_file, "*");

  ConvLayerParameters cp{max_file, "conv0"};
  LOG(INFO) << "Num fraction bits: " << cp.dfe.num_frac_bits << '\n';

  std::vector<ConvLayerParameters> cps;
  for (int i = 0; i < NUM_LAYER; ++i)
    cps.push_back(ConvLayerParameters{max_file, "conv" + std::to_string(i)});

  // for (uint64_t i = 0; i < ifmap_num_elems; i ++)
  //   ifmap[i] = (rand() % 10) - 5;
  // for (uint64_t i = 0; i < coeff_0_num_elems; i ++)
  //   coeff_0[i] = (rand() % 10) - 5;

  printf("Initializing arrays ...\n");

  float max_val = 1.1, min_val = -1.1, input_scale = 1;
  auto input =
      CreateRandomArray<float>(cp.dfe.TC * cp.dfe.TH * cp.dfe.TW * batch_size,
                               input_scale * min_val, input_scale * max_val);
  auto input_dfe = FloatToFixed<data_t>(input, cp.dfe.num_frac_bits);

  std::vector<std::vector<float>> weights_dw_list, weights_pw_list;
  for (int i = 0; i < NUM_LAYER; ++i) {
    weights_dw_list.push_back(
        ReadDataFile(data_file, "conv" + std::to_string(i) + "_dw"));
    weights_pw_list.push_back(
        ReadDataFile(data_file, "conv" + std::to_string(i) + "_pw"));
  }

  DepthwiseSeparableConvLayer_actions_t actions;
  actions.param_batch_size = batch_size;

  // #ifndef USE_DRAM
  // #error "Not supported"
  //   uint64_t ifmap_num_elems = H * W * C * batch_size;
  //   uint64_t ofmap_num_elems = (H - K + 1) * (W - K + 1) * F * batch_size;

  //   auto ifmap = random_initialize<data_t>(ifmap_num_elems, 100);
  //   auto ofmap = create_array<data_t>(ofmap_num_elems);
  //   auto ofmap_golden = create_array<data_t>(ofmap_num_elems);

  //   actions.instream_ifmap = (const data_t *)ifmap;
  //   actions.outstream_ofmap = ofmap;

  // #ifndef DEPTHWISE_SEPARABLE_V2
  //   uint64_t depthwise_coeff_num_elems = C * K * K * batch_size;
  //   uint64_t pointwise_coeff_num_elems = C * F * batch_size;
  //   data_t *depthwise_coeff_0 =
  //       (data_t *)malloc(sizeof(data_t) * depthwise_coeff_num_elems);
  //   data_t *pointwise_coeff_0 =
  //       (data_t *)malloc(sizeof(data_t) * pointwise_coeff_num_elems);
  //   actions.instream_depthwise_coeff_0 = (const data_t *)depthwise_coeff_0;
  //   actions.instream_pointwise_coeff_0 = (const data_t *)pointwise_coeff_0;
  // #else
  //   uint64_t coeff_num_elems = C * K * K * (1 + F) * batch_size;
  //   auto coeff_0 = random_initialize<data_t>(coeff_num_elems, 100);
  //   actions.instream_coeff_0 = (const data_t *)coeff_0;

  //   dump_array("coeff.txt", coeff_0, coeff_num_elems);
  // #endif

  // #else

  // #endif

  std::vector<std::vector<data_t>> weights_dw_dfe_list, weights_pw_dfe_list;
  for (int i = 0; i < NUM_LAYER; ++i) {
    weights_dw_dfe_list.push_back(
        FloatToFixed<data_t>(weights_dw_list[i], cps[i].dfe.num_frac_bits));
    weights_pw_dfe_list.push_back(
        FloatToFixed<data_t>(weights_pw_list[i], cps[i].dfe.num_frac_bits));
  }

  // auto output_cpu = std::vector<data_t>(cp.F * cp.getOutputHeight() *
  //                                       cp.getOutputWidth() * batch_size);
  auto output_dfe =
      std::vector<data_t>(cps.back().F * cps.back().getOutputHeight() *
                          cps.back().getOutputWidth() * batch_size);

  printf("Running golden function ...\n");
  // depthwise_separable_conv_layer(ifmap, coeff_0, ofmap_golden, H, W, C, F, K,
  //                                batch_size);
  auto dummy_bias = std::vector<data_t>();

  std::vector<std::vector<data_t>> output_cpu_list;
  output_cpu_list.push_back(input_dfe);
  for (int i = 0; i < NUM_LAYER; ++i) {
    ConvLayerParameters cp = cps[i];
    LOG(INFO) << cp.C << " x " << cp.H << " x " << cp.W << " -> " << cp.F
              << " x " << cp.getOutputHeight() << " x " << cp.getOutputWidth()
              << '\n';

    auto output_cpu = std::vector<data_t>(cp.F * cp.getOutputHeight() *
                                          cp.getOutputWidth() * batch_size);
    DepthwiseSeparableConvLayerCpuBatched<data_t>(
        output_cpu_list.back(), weights_dw_dfe_list[i], weights_pw_dfe_list[i],
        dummy_bias, output_cpu, batch_size, cp.H, cp.W, cp.C, cp.F, cp.K, cp.P,
        cp.S,
        /*use_bias=*/false,
        /*use_fixed_point=*/true, cp.dfe.num_frac_bits);
    output_cpu_list.push_back(output_cpu);
  }

  input_dfe = ReorderInput(input_dfe, cp, batch_size);

#ifdef USE_DRAM
  constexpr size_t num_bytes_per_burst = 384;
  size_t base_addr = 0;

  BurstAlign(input_dfe, num_bytes_per_burst * cp.dfe.PC * cp.dfe.PK);
  LOG(INFO) << "Tiled input size (burst aligned): " << input_dfe.size();
  BurstAlign(output_dfe, num_bytes_per_burst * cp.dfe.PF * cp.dfe.PK);
  LOG(INFO) << "Tiled output size (burst aligned): " << output_dfe.size();

  base_addr = WriteDRAM<data_t, Dfe>(input_dfe, base_addr, engine);
#endif

  printf("Running ...\n");
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < (int)num_iters; i++)
    DepthwiseSeparableConvLayer_run(engine, &actions);
  end = std::chrono::system_clock::now();
  printf("Done\n");

#ifdef USE_DRAM
  ReadDRAM<data_t, Dfe>(output_dfe, base_addr, engine);
#endif

  output_dfe = ReorderOutput(output_dfe, cp, batch_size);

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() / 1 << "s\n";
  // uint64_t num_ops = (((H - K + 1) * (W - K + 1) * C * K * K) +
  //                     ((H - K + 1) * (W - K + 1) * C * F)) *
  //                    2;
  // uint64_t num_conv_ops = (H - K + 1) * (W - K + 1) * C * F * K * K * 2;

  // std::cout << "GOP/s: "
  //           << num_ops * batch_size * 1e-9 / elapsed_seconds.count() *
  //           num_iters
  //           << std::endl;
  // std::cout << "GOP/s: "
  //           << num_conv_ops * batch_size * 1e-9 / elapsed_seconds.count() *
  //                  num_iters
  //           << " (CONV)" << std::endl;

  // for (int i = 0; i < 10; i++) printf("ofmap[%5d] = %d\n", i, ofmap[i]);
  // printf("Golden:\n");
  // for (int i = 0; i < 10; i++) printf("ofmap[%5d] = %d\n", i,
  // ofmap_golden[i]);

  // printf("Running test ...\n");
  // for (int i = 0; i < ofmap_num_elems; i++)
  //   if (ofmap[i] != ofmap_golden[i]) {
  //     fprintf(stderr, "ofmap doesn't matched at %d: %d != %d\n", i, ofmap[i],
  //             ofmap_golden[i]);
  //     exit(1);
  //   }
  auto &output_cpu = output_cpu_list.back();
  LOG(INFO) << "Examine results ...\n";
  for (int i = 0; i < 5; ++i)
    LOG(INFO) << "output_cpu[" << i << "] = "
              << FixedToFloat<data_t>(output_cpu[i], cp.dfe.num_frac_bits)
              << '\n';

  for (int i = 0; i < 5; ++i)
    LOG(INFO) << "output_dfe[" << i << "] = "
              << FixedToFloat<data_t>(output_dfe[i], cp.dfe.num_frac_bits)
              << '\n';

  auto output_cpu_float =
      FixedToFloat<data_t>(output_cpu, cp.dfe.num_frac_bits);
  auto output_dfe_float =
      FixedToFloat<data_t>(output_dfe, cp.dfe.num_frac_bits);

  const int16_t mask = (1 << (cp.dfe.num_frac_bits + 1)) - 1;
  LOG(INFO) << "Evaluating difference ...";
  LOG(INFO) << "Mask: " << std::setw(8) << std::showbase << std::setfill('0')
            << std::hex << mask;

  uint64_t num_failed = 0;
  double total_diff = 0.0f;
  for (uint64_t i = 0; i < output_cpu.size(); i++) {
    auto diff = std::abs(output_cpu[i] - output_dfe[i]);
    auto float_diff = std::abs(output_cpu_float[i] - output_dfe_float[i]);
    total_diff += float_diff;
    if ((static_cast<int16_t>(diff) | mask) != mask) {
      if (num_failed < 10)
        fprintf(stderr, "Result mis-matched at %6ld: cpu %10.6f dfe %10.6f\n",
                i, output_cpu_float[i], output_dfe_float[i]);
      num_failed++;
      // exit(1);
    }
  }

  LOG(INFO) << "num_failed: " << num_failed << " total: " << output_cpu.size()
            << " avg_diff: " << (total_diff / output_cpu.size()) << '\n';
  // printf("Test PASSED!\n");

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
