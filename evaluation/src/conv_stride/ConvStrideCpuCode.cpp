#include <getopt.h>
#include <glog/logging.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include "Maxfiles.h"
#include "maxdeep/layers.h"

// Can we reflect the types from Maxfiles.h?
typedef int16_t data_t;

struct Dfe {
  typedef ConvStride_actions_t dfe_run_actions_t;
  typedef ConvStride_dramRead_actions_t dram_read_actions_t;
  typedef ConvStride_dramWrite_actions_t dram_write_actions_t;

  static void ReadDRAM(max_engine_t *engine, dram_read_actions_t *actions) {
    ConvStride_dramRead_run(engine, actions);
  }
  static void WriteDRAM(max_engine_t *engine, dram_write_actions_t *actions) {
    ConvStride_dramWrite_run(engine, actions);
  }
};

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  char c;

  uint64_t batch_size = 1;
  uint64_t num_iters = 1;
  bool cpu_sim = false;
  while ((c = getopt(argc, argv, "i:n:c")) != -1) switch (c) {
      case 'i':
        num_iters = atoi(optarg);
        break;
      case 'n':
        batch_size = atoi(optarg);
        break;
      case 'c':
        cpu_sim = true;
        break;
      default:
        exit(1);
    }

  srand(42);

  max_file_t *max_file = ConvStride_init();
  max_engine_t *engine = max_load(max_file, "*");

  // load constants

  ConvLayerParameters cp0{max_file, "conv0"};

  // The input size is the same as what the processor support.
  float max_val = 5, min_val = -5;
  auto input = CreateRandomArray<float>(cp0.C * cp0.H * cp0.W * batch_size,
                                        min_val / cp0.F, max_val / cp0.F);
  auto weights_0 =
      CreateRandomArray<float>(cp0.F * cp0.C * cp0.K * cp0.K, min_val, max_val);

  // assuming the processing type is fixed point.
  auto input_dfe = FloatToFixed<data_t>(input, cp0.dfe.num_frac_bits);
  auto weights_0_dfe = FloatToFixed<data_t>(weights_0, cp0.dfe.num_frac_bits);
  auto output_dfe = std::vector<data_t>(cp0.F * cp0.getOutputHeight() *
                                        cp0.getOutputWidth() * batch_size);
  auto dummy_bias = std::vector<data_t>();

  LOG(INFO) << "Created and converted input and weights\n";

  // CPU execution.
  std::vector<data_t> output_0_cpu;
  if (cpu_sim) {
    LOG(INFO) << "Run ConvLayerCpu ...\n";
    output_0_cpu = std::vector<data_t>(cp0.F * cp0.getOutputHeight() *
                                       cp0.getOutputWidth() * batch_size);
    LOG(INFO) << cp0.C << " x " << cp0.H << " x " << cp0.W << " -> " << cp0.F
              << " x " << cp0.getOutputHeight() << " x " << cp0.getOutputWidth()
              << " PAD = " << cp0.P << '\n';
    ConvLayerCpuBatched<data_t>(
        input_dfe, weights_0_dfe, dummy_bias, output_0_cpu, batch_size, cp0.H,
        cp0.W, cp0.C, cp0.F, cp0.K, cp0.P, cp0.S,
        /*use_bias=*/false,
        /*use_fixed_point=*/true, cp0.dfe.num_frac_bits);
    LOG(INFO) << "Done\n";
  }

  input_dfe = ReorderInput(input_dfe, cp0, batch_size);

  ConvStride_actions_t actions;
  actions.param_batch_size = batch_size;

  assert(cp0.dfe.coeff_on_chip);
  // Split weights into the FMems
  double **ptr = reinterpret_cast<double **>(&(actions.param_batch_size) + 1);
  ptr = SplitCoeffAndAssign<float>(ptr, weights_0.data(), cp0);
  LOG(INFO) << "Initialized coefficients from weights_0_dfe, ptr = " << ptr
            << '\n';
  LOG(INFO) << "Initialized actions\n";

#ifndef USE_DRAM
  assert(false);
#else
  constexpr size_t num_bytes_per_burst = 384;
  size_t base_addr = 0;

  BurstAlign(input_dfe, num_bytes_per_burst * cp0.dfe.PC * cp0.dfe.PK);
  LOG(INFO) << "Tiled input size (burst aligned): " << input_dfe.size();
  BurstAlign(output_dfe, num_bytes_per_burst * cp0.dfe.PF * cp0.dfe.PK);
  LOG(INFO) << "Tiled output size (burst aligned): " << output_dfe.size();

  base_addr = WriteDRAM<data_t, Dfe>(input_dfe, base_addr, engine);
#endif
  std::chrono::time_point<std::chrono::system_clock> start, end;
  actions.param_batch_size = 1;
  LOG(INFO) << "Initialise ...\n";
  start = std::chrono::system_clock::now();
  // if (batch_size > 1) ConvStride_run(engine, &actions);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> init_time = end - start;
  std::cout << "elapsed time: " << init_time.count() / 1 << "s\n";

  LOG(INFO) << "Running ...\n";
  actions.param_batch_size = batch_size;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < (int)num_iters; i++) ConvStride_run(engine, &actions);
  end = std::chrono::system_clock::now();
  printf("Done\n");

  std::chrono::duration<double> total_time = end - start;
  std::cout << "elapsed time: " << total_time.count() / 1 << "s\n";

  auto elapsed_time = batch_size == 1
                          ? total_time.count()
                          : ((total_time.count() - init_time.count()) *
                             ((double)batch_size / (batch_size - 1)));

#ifdef USE_DRAM
  ReadDRAM<data_t, Dfe>(output_dfe, base_addr, engine);
#endif
  output_dfe = ReorderOutput(output_dfe, cp0, batch_size);

  if (cpu_sim) {
    LOG(INFO) << "Examine results ...\n";
    for (int i = 0; i < 5; ++i)
      LOG(INFO) << "output_1_cpu[" << i << "] = "
                << FixedToFloat<data_t>(output_0_cpu[i], cp0.dfe.num_frac_bits)
                << '\n';

    for (int i = 0; i < 5; ++i)
      LOG(INFO) << "output_dfe[" << i << "] = "
                << FixedToFloat<data_t>(output_dfe[i], cp0.dfe.num_frac_bits)
                << '\n';

    auto output_cpu_float =
        FixedToFloat<data_t>(output_0_cpu, cp0.dfe.num_frac_bits);
    auto output_dfe_float =
        FixedToFloat<data_t>(output_dfe, cp0.dfe.num_frac_bits);

    const int16_t mask = (1 << (cp0.dfe.num_frac_bits + 1)) - 1;
    LOG(INFO) << "Evaluating difference ...";
    LOG(INFO) << "Mask: " << std::setw(8) << std::showbase << std::setfill('0')
              << std::hex << mask;

    uint64_t num_failed = 0;
    double total_diff = 0.0f;
    for (uint64_t i = 0; i < output_0_cpu.size(); i++) {
      auto diff = std::abs(output_0_cpu[i] - output_dfe[i]);
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

    LOG(INFO) << "num_failed: " << num_failed
              << " total: " << output_0_cpu.size()
              << " avg_diff: " << (total_diff / output_0_cpu.size()) << '\n';
  }

  uint64_t num_ops = 0;
  // conv0
  num_ops += cp0.getOutputHeight() * cp0.getOutputWidth() * cp0.C * cp0.F *
             cp0.K * cp0.K * 2;

  std::cout << "GOP/s: "
            << num_ops * batch_size * 1e-9 / (elapsed_time / num_iters)
            << std::endl;

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
