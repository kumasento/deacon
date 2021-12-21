#include <getopt.h>
#include <glog/logging.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

// #define TRACE

#include "Maxfiles.h"
#include "maxdeep/layers.h"

// Can we reflect the types from Maxfiles.h?
#if BIT_WIDTH == 16
typedef int16_t data_t;
#elif BIT_WIDTH == 8
typedef int8_t data_t;
#endif

struct Dfe {
  typedef Bottleneck_actions_t dfe_run_actions_t;
  typedef Bottleneck_dramRead_actions_t dram_read_actions_t;
  typedef Bottleneck_dramWrite_actions_t dram_write_actions_t;

  static void ReadDRAM(max_engine_t *engine, dram_read_actions_t *actions) {
    Bottleneck_dramRead_run(engine, actions);
  }
  static void WriteDRAM(max_engine_t *engine, dram_write_actions_t *actions) {
    Bottleneck_dramWrite_run(engine, actions);
  }
};

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  char c;

  uint64_t batch_size = 1;
  uint64_t num_iters = 1;
  while ((c = getopt(argc, argv, "i:n:")) != -1) switch (c) {
      case 'i':
        num_iters = atoi(optarg);
        break;
      case 'n':
        batch_size = atoi(optarg);
        break;
      default:
        exit(1);
    }

  srand(42);

  max_file_t *max_file = Bottleneck_init();
  max_engine_t *engine = max_load(max_file, "*");

  // load constants

  ConvLayerParameters cp0{max_file, "conv0"};
  ConvLayerParameters cp1{max_file, "conv1"};
  ConvLayerParameters cp2{max_file, "conv2"};

  cp0.dump();
  cp1.dump();
  cp2.dump();

  // generate input data
  // The input size is the same as what the processor support.
  float max_val = 1.1, min_val = -1.1, input_scale = 1;
  auto input = CreateRandomArray<float>(
      cp0.dfe.TC * cp0.dfe.TH * cp0.dfe.TW * batch_size, input_scale * min_val,
      input_scale * max_val);
  auto weights_0 = CreateRandomArray<float>(
      cp0.dfe.TF * cp0.dfe.TC * cp0.dfe.K * cp0.dfe.K, min_val, max_val);
  auto weights_1 = CreateRandomArray<float>(
      cp1.dfe.TF * cp1.dfe.TC * cp1.dfe.K * cp1.dfe.K, min_val, max_val);
  auto weights_2 = CreateRandomArray<float>(
      cp2.dfe.TF * cp2.dfe.TC * cp2.dfe.K * cp2.dfe.K, min_val, max_val);
  // auto input = CreateConstantArray<float>(
  //     cp0.dfe.TC * cp0.dfe.TH * cp0.dfe.TW * batch_size, 0.1);
  // auto weights_0 = CreateConstantArray<float>(
  //     cp0.dfe.TF * cp0.dfe.TC * cp0.dfe.K * cp0.dfe.K, 1);
  // auto weights_1 = CreateConstantArray<float>(
  //     cp1.dfe.TF * cp1.dfe.TC * cp1.dfe.K * cp1.dfe.K, 1);
  // auto weights_2 = CreateConstantArray<float>(
  //     cp2.dfe.TF * cp2.dfe.TC * cp2.dfe.K * cp2.dfe.K, 1);

  // assuming the processing type is fixed point.
  auto input_dfe = FloatToFixed<data_t>(input, cp0.dfe.num_frac_bits);
#if WBW == 2
  auto weights_0_dfe =
      FloatToFixed<data_t>(ToTernary<float>(weights_0), cp0.dfe.num_frac_bits);
  auto weights_1_dfe =
      FloatToFixed<data_t>(ToTernary<float>(weights_1), cp1.dfe.num_frac_bits);
  auto weights_2_dfe =
      FloatToFixed<data_t>(ToTernary<float>(weights_2), cp2.dfe.num_frac_bits);
#else
  auto weights_0_dfe = FloatToFixed<data_t>(weights_0, cp0.dfe.num_frac_bits);
  auto weights_1_dfe = FloatToFixed<data_t>(weights_1, cp1.dfe.num_frac_bits);
  auto weights_2_dfe = FloatToFixed<data_t>(weights_2, cp2.dfe.num_frac_bits);
#endif
  auto output_0_cpu = std::vector<data_t>(cp0.F * cp0.getOutputHeight() *
                                          cp0.getOutputWidth() * batch_size);
  auto output_1_cpu = std::vector<data_t>(cp1.F * cp1.getOutputHeight() *
                                          cp1.getOutputWidth() * batch_size);
  auto output_2_cpu = std::vector<data_t>(cp2.F * cp2.getOutputHeight() *
                                          cp2.getOutputWidth() * batch_size);
  auto output_dfe = std::vector<data_t>(cp2.F * cp2.getOutputHeight() *
                                        cp2.getOutputWidth() * batch_size);
  auto dummy_bias = std::vector<data_t>();

  LOG(INFO) << "Created and converted input and weights\n";

  // CPU execution.
  LOG(INFO) << "Run ConvLayerCpu ...\n";
  LOG(INFO) << cp0.C << " x " << cp0.H << " x " << cp0.W << " -> " << cp0.F
            << " x " << cp0.getOutputHeight() << " x " << cp0.getOutputWidth()
            << '\n';
  ConvLayerCpuBatched<data_t>(input_dfe, weights_0_dfe, dummy_bias,
                              output_0_cpu, batch_size, cp0.H, cp0.W, cp0.C,
                              cp0.F, cp0.K, cp0.P, cp0.S,
                              /*use_bias=*/false,
                              /*use_fixed_point=*/true, cp0.dfe.num_frac_bits);
  LOG(INFO) << cp1.C << " x " << cp1.H << " x " << cp1.W << " -> " << cp1.F
            << " x " << cp1.getOutputHeight() << " x " << cp1.getOutputWidth()
            << '\n';
  ConvLayerCpuBatched<data_t>(output_0_cpu, weights_1_dfe, dummy_bias,
                              output_1_cpu, batch_size, cp1.H, cp1.W, cp1.C,
                              cp1.F, cp1.K, cp1.P, cp1.S,
                              /*use_bias=*/false,
                              /*use_fixed_point=*/true, cp1.dfe.num_frac_bits);
  LOG(INFO) << cp2.C << " x " << cp2.H << " x " << cp2.W << " -> " << cp2.F
            << " x " << cp2.getOutputHeight() << " x " << cp2.getOutputWidth()
            << '\n';
  ConvLayerCpuBatched<data_t>(output_1_cpu, weights_2_dfe, dummy_bias,
                              output_2_cpu, batch_size, cp2.H, cp2.W, cp2.C,
                              cp2.F, cp2.K, cp2.P, cp2.S,
                              /*use_bias=*/false,
                              /*use_fixed_point=*/true, cp2.dfe.num_frac_bits);
  for (int i = 0; i < output_2_cpu.size(); ++i) {
    output_2_cpu[i] = FixedPointAdd<data_t>(input_dfe[i], output_2_cpu[i]);
  }
  LOG(INFO) << "Done\n";

  // DumpArray<data_t>("input.txt", input_dfe.data(), input_dfe.size());
  // DumpArray<data_t>("output_0_cpu.txt", output_0_cpu.data(),
  //                   output_0_cpu.size(), cp0.dfe.num_frac_bits);
  // DumpArray<data_t>("output_1_cpu.txt", output_1_cpu.data(),
  //                   output_1_cpu.size(), cp1.dfe.num_frac_bits);
  // DumpArray<data_t>("output_2_cpu.txt", output_2_cpu.data(),
  //                   output_2_cpu.size(), cp2.dfe.num_frac_bits);

  input_dfe = ReorderInput(input_dfe, cp0, batch_size);
  // Configure DFE run.
  Bottleneck_actions_t actions;
  actions.param_batch_size = batch_size;

  assert(cp0.dfe.coeff_on_chip && cp1.dfe.coeff_on_chip &&
         cp2.dfe.coeff_on_chip);
  // Split weights into the FMems
  LOG(INFO) << "BIT_WIDTH = " << BIT_WIDTH << " WBW = " << WBW << '\n';
#if WBW <= 8
#if WBW == 2
  weights_0_dfe = ToTernaryUnsigned<data_t, data_t>(weights_0_dfe);
  weights_1_dfe = ToTernaryUnsigned<data_t, data_t>(weights_1_dfe);
  weights_2_dfe = ToTernaryUnsigned<data_t, data_t>(weights_2_dfe);
#endif

#if 0
  LOG(INFO) << "Split into uint64_t ...";
  uint64_t **ptr =
      reinterpret_cast<uint64_t **>(&(actions.param_batch_size) + 1);
  ptr = SplitCoeffAndAssign<data_t, uint64_t>(ptr, weights_0_dfe.data(), cp0);
  LOG(INFO) << "Initialized coefficients from weights_0, ptr = " << ptr << '\n';
  ptr = SplitCoeffAndAssign<data_t, uint64_t>(ptr, weights_1_dfe.data(), cp1);
  LOG(INFO) << "Initialized coefficients from weights_1, ptr = " << ptr << '\n';
  ptr = SplitCoeffAndAssign<data_t, uint64_t>(ptr, weights_2_dfe.data(), cp2);
  LOG(INFO) << "Initialized coefficients from weights_2, ptr = " << ptr << '\n';
#endif

  int8_t *test2 = (int8_t *)malloc(
      std::max(std::max(weights_0.size(), weights_1.size()), weights_2.size()));

  Bottleneck_initCoeff_actions_t init_coeff_actions;
  init_coeff_actions.ticks_conv0 = (uint64_t)cp0.F * cp0.C * cp0.K * cp0.K;
  init_coeff_actions.ticks_conv1 = 0;
  init_coeff_actions.ticks_conv2 = 0;
  init_coeff_actions.inscalar_conv0_init_coeff = 1;
  init_coeff_actions.inscalar_conv1_init_coeff = 0;
  init_coeff_actions.inscalar_conv2_init_coeff = 0;
  init_coeff_actions.instream_init_coeff = weights_0_dfe.data();
  init_coeff_actions.outstream_init_coeff_out = test2;
  init_coeff_actions.param_init_coeff_num_elems = weights_0_dfe.size();
  init_coeff_actions.routing_string =
      "init_coeff_demux -> conv0, init_coeff_mux -> conv0";

  Bottleneck_initCoeff_run(engine, &init_coeff_actions);
  LOG(INFO) << "initCoeff for conv0. size: " << weights_0_dfe.size()
            << " ticks: " << init_coeff_actions.ticks_conv0 << '\n';

  init_coeff_actions.ticks_conv0 = 0;
  init_coeff_actions.ticks_conv1 = (uint64_t)cp1.F * cp1.C * cp1.K * cp1.K;
  init_coeff_actions.ticks_conv2 = 0;
  init_coeff_actions.inscalar_conv0_init_coeff = 0;
  init_coeff_actions.inscalar_conv1_init_coeff = 1;
  init_coeff_actions.inscalar_conv2_init_coeff = 0;
  init_coeff_actions.instream_init_coeff = weights_1_dfe.data();
  init_coeff_actions.outstream_init_coeff_out = test2;
  init_coeff_actions.param_init_coeff_num_elems = weights_1_dfe.size();
  init_coeff_actions.routing_string =
      "init_coeff_demux -> conv1, init_coeff_mux -> conv1";

  Bottleneck_initCoeff_run(engine, &init_coeff_actions);
  LOG(INFO) << "initCoeff for conv1. size: " << weights_1_dfe.size()
            << " ticks: " << init_coeff_actions.ticks_conv1 << '\n';

  init_coeff_actions.ticks_conv0 = 0;
  init_coeff_actions.ticks_conv1 = 0;
  init_coeff_actions.ticks_conv2 = (uint64_t)cp2.F * cp2.C * cp2.K * cp2.K;
  init_coeff_actions.inscalar_conv0_init_coeff = 0;
  init_coeff_actions.inscalar_conv1_init_coeff = 0;
  init_coeff_actions.inscalar_conv2_init_coeff = 1;
  init_coeff_actions.instream_init_coeff = weights_2_dfe.data();
  init_coeff_actions.outstream_init_coeff_out = test2;
  init_coeff_actions.param_init_coeff_num_elems = weights_2_dfe.size();
  init_coeff_actions.routing_string =
      "init_coeff_demux -> conv2, init_coeff_mux -> conv2";

  Bottleneck_initCoeff_run(engine, &init_coeff_actions);
  LOG(INFO) << "initCoeff for conv2. size: " << weights_2_dfe.size()
            << " ticks: " << init_coeff_actions.ticks_conv2 << '\n';
#else
  double **ptr = reinterpret_cast<double **>(&(actions.param_batch_size) + 1);
  ptr = SplitCoeffAndAssign<float>(ptr, weights_0.data(), cp0);
  LOG(INFO) << "Initialized coefficients from weights_0, ptr = " << ptr << '\n';
  ptr = SplitCoeffAndAssign<float>(ptr, weights_1.data(), cp1);
  LOG(INFO) << "Initialized coefficients from weights_1, ptr = " << ptr << '\n';
  ptr = SplitCoeffAndAssign<float>(ptr, weights_2.data(), cp2);
  LOG(INFO) << "Initialized coefficients from weights_2, ptr = " << ptr << '\n';
#endif

  // std::string routing_string = "conv0_fanout -> conv0, conv0_fanout ->
  // conv2"; actions.routing_string = routing_string.c_str();

  LOG(INFO) << "Initialized actions\n";

#ifndef USE_DRAM
  actions.instream_coeff_0 = (const data_t *)coeff_0;
  actions.instream_coeff_1 = (const data_t *)coeff_1;
  actions.instream_ifmap = (const data_t *)ifmap;
  actions.outstream_ofmap = ofmap;
#else
  constexpr size_t num_bytes_per_burst = 384;
  size_t base_addr = 0;

  BurstAlign(input_dfe, num_bytes_per_burst * cp0.dfe.PC * cp0.dfe.PK);
  LOG(INFO) << "Tiled input size (burst aligned): " << input_dfe.size();
  BurstAlign(output_dfe, num_bytes_per_burst * cp2.dfe.PF * cp2.dfe.PK);
  LOG(INFO) << "Tiled output size (burst aligned): " << output_dfe.size();

  base_addr = WriteDRAM<data_t, Dfe>(input_dfe, base_addr, engine);
#endif

  printf("Running ...\n");
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < (int)num_iters; i++) Bottleneck_run(engine, &actions);
  end = std::chrono::system_clock::now();
  printf("Done\n");

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() / 1 << "s\n";

#ifdef USE_DRAM
  ReadDRAM<data_t, Dfe>(output_dfe, base_addr, engine);
#endif

  output_dfe = ReorderOutput(output_dfe, cp2, batch_size);
  LOG(INFO) << "Examine results ...\n";
  for (int i = 0; i < 5; ++i)
    LOG(INFO) << "output_2_cpu[" << i << "] = "
              << FixedToFloat<data_t>(output_2_cpu[i], cp2.dfe.num_frac_bits)
              << '\n';

  for (int i = 0; i < 5; ++i)
    LOG(INFO) << "output_dfe[" << i << "] = "
              << FixedToFloat<data_t>(output_dfe[i], cp2.dfe.num_frac_bits)
              << '\n';

  auto output_cpu_float =
      FixedToFloat<data_t>(output_2_cpu, cp2.dfe.num_frac_bits);
  auto output_dfe_float =
      FixedToFloat<data_t>(output_dfe, cp2.dfe.num_frac_bits);

  const int16_t mask = (1 << (cp2.dfe.num_frac_bits + 1)) - 1;
  LOG(INFO) << "Evaluating difference ...";
  LOG(INFO) << "Mask: " << std::setw(8) << std::showbase << std::setfill('0')
            << std::hex << mask;

  uint64_t num_failed = 0;
  double total_diff = 0.0f;
  for (uint64_t i = 0; i < output_2_cpu.size(); i++) {
    auto diff = std::abs(output_2_cpu[i] - output_dfe[i]);
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

  LOG(INFO) << "num_failed: " << num_failed << " total: " << output_1_cpu.size()
            << " avg_diff: " << (total_diff / output_1_cpu.size()) << '\n';

  uint64_t num_ops = 0;
  // conv0
  num_ops += cp0.H * cp0.W * cp0.C * cp0.F * cp0.K * cp0.K * 2;
  // conv1
  num_ops += cp1.H * cp1.W * cp1.C * cp1.F * cp1.K * cp1.K * 2;
  // conv2
  num_ops += cp2.H * cp2.W * cp2.C * cp2.F * cp2.K * cp2.K * 2;

  std::cout << "GOP/s: "
            << num_ops * batch_size * 1e-9 /
                   (elapsed_seconds.count() / num_iters)
            << std::endl;

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
