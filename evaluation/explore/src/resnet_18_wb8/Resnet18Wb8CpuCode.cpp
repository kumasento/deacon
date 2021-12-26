#include <getopt.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <unordered_map>

#include "Maxfiles.h"
#include "maxdeep/layers.h"
#include "maxdeep/utils.h"

typedef int8_t data_t;

struct Dfe {
  typedef Resnet18Wb8_actions_t dfe_run_actions_t;
  typedef Resnet18Wb8_dramRead_actions_t dram_read_actions_t;
  typedef Resnet18Wb8_dramWrite_actions_t dram_write_actions_t;

  static void ReadDRAM(max_engine_t *engine, dram_read_actions_t *actions) {
    Resnet18Wb8_dramRead_run(engine, actions);
  }
  static void WriteDRAM(max_engine_t *engine, dram_write_actions_t *actions) {
    Resnet18Wb8_dramWrite_run(engine, actions);
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
  bool run_golden = false;
  while ((c = getopt(argc, argv, "i:n:gf:")) != -1) {
    switch (c) {
      case 'i':
        num_iters = atoi(optarg);
        break;
      case 'n':
        batch_size = atoi(optarg);
        break;
      case 'g':
        run_golden = true;
        break;
      case 'f':
        data_file = std::string(optarg);
        break;
      default:
        exit(1);
    }
  }

  max_file_t *max_file = Resnet18Wb8_init();
  max_engine_t *engine = max_load(max_file, "*");

  std::vector<ConvLayerParameters> cps;
  cps.push_back(ConvLayerParameters(max_file, "conv0"));
  cps.push_back(ConvLayerParameters(max_file, "pool0"));
  cps.push_back(ConvLayerParameters(max_file, "conv1"));
  cps.push_back(ConvLayerParameters(max_file, "conv2"));
  cps.push_back(ConvLayerParameters(max_file, "conv3"));
  cps.push_back(ConvLayerParameters(max_file, "conv4"));
  cps.push_back(ConvLayerParameters(max_file, "conv5"));
  cps.push_back(ConvLayerParameters(max_file, "shortcut2"));
  cps.push_back(ConvLayerParameters(max_file, "conv6"));
  cps.push_back(ConvLayerParameters(max_file, "conv7"));
  cps.push_back(ConvLayerParameters(max_file, "conv8"));
  cps.push_back(ConvLayerParameters(max_file, "conv9"));
  cps.push_back(ConvLayerParameters(max_file, "shortcut3"));
  cps.push_back(ConvLayerParameters(max_file, "conv10"));
  cps.push_back(ConvLayerParameters(max_file, "conv11"));
  cps.push_back(ConvLayerParameters(max_file, "conv12"));
  cps.push_back(ConvLayerParameters(max_file, "conv13"));
  cps.push_back(ConvLayerParameters(max_file, "shortcut4"));
  cps.push_back(ConvLayerParameters(max_file, "conv14"));
  cps.push_back(ConvLayerParameters(max_file, "conv15"));
  cps.push_back(ConvLayerParameters(max_file, "conv16"));

  /* Generate input data */
  float max_val = 1.1, min_val = -1.1, input_scale = 1;
  auto input = CreateRandomArray<float>(
      cps.front().dfe.TC * cps.front().dfe.TH * cps.front().dfe.TW * batch_size,
      input_scale * min_val, input_scale * max_val);
  LOG(INFO) << "num frac bits: " << cps.front().dfe.num_frac_bits << '\n';
  auto input_dfe = FloatToFixed<data_t>(input, cps.front().dfe.num_frac_bits);

  for (int i = 0; i < 3; ++i)
    LOG(INFO) << "input[" << i << "] = " << input[i] << "  dfe = "
              << FixedToFloat<data_t>(input_dfe[i],
                                      cps.front().dfe.num_frac_bits)
              << '\n';

  /** Run golden test. */
  std::vector<std::vector<data_t>> buffers;
  if (run_golden) {
    LOG(INFO) << "Running golden function ...\n";

    auto dummy_bias = std::vector<data_t>();
    buffers.push_back(input_dfe);

    std::unordered_map<std::string, int> name_to_output;

    for (int i = 0; i < cps.size(); ++i) {
      ConvLayerParameters cp = cps[i];
      LOG(INFO) << std::setw(15) << std::setfill(' ') << cp.dfe.TYPE << " "
                << cp.C << " x " << cp.H << " x " << cp.W << " -> " << cp.F
                << " x " << cp.getOutputHeight() << " x " << cp.getOutputWidth()
                << '\n';

      auto output_cpu = std::vector<data_t>(cp.F * cp.getOutputHeight() *
                                            cp.getOutputWidth() * batch_size);
      if (cp.dfe.TYPE == "STANDARD") {
        std::vector<float> weights = ReadDataFile(data_file, cp.dfe.name);
        std::vector<data_t> weights_dfe =
            FloatToFixed<data_t>(weights, cp.dfe.num_frac_bits);

        ConvLayerCpuBatched<data_t>(
            buffers.back(), weights_dfe, dummy_bias, output_cpu, batch_size,
            cp.H, cp.W, cp.C, cp.F, cp.K, cp.P, cp.S,
            /*use_bias=*/false,
            /*use_fixed_point=*/true, cp.dfe.num_frac_bits);
      } else if (cp.dfe.TYPE == "DEPTHWISE_SEPARABLE") {
        std::vector<float> weights_dw, weights_pw;
        weights_dw = ReadDataFile(data_file, cp.dfe.name + "_dw");
        weights_pw = ReadDataFile(data_file, cp.dfe.name + "_pw");

        std::vector<data_t> weights_dw_dfe, weights_pw_dfe;
        weights_dw_dfe = FloatToFixed<data_t>(weights_dw, cp.dfe.num_frac_bits);
        weights_pw_dfe = FloatToFixed<data_t>(weights_pw, cp.dfe.num_frac_bits);

        DepthwiseSeparableConvLayerCpuBatched<data_t>(
            buffers.back(), weights_dw_dfe, weights_pw_dfe, dummy_bias,
            output_cpu, batch_size, cp.H, cp.W, cp.C, cp.F, cp.K, cp.P, cp.S,
            /*use_bias=*/false,
            /*use_fixed_point=*/true, cp.dfe.num_frac_bits);
      } else {
        LOG(ERROR) << "Doesn't recognize type: " << cp.dfe.TYPE << '\n';
      }

      for (int j = 0; j < cp.dfe.NUM_OUTPUTS; ++j) {
        std::string key = cp.dfe.name;
        if (j != 0) key += "_" + std::to_string(j);
        if (cp.dfe.OUTPUTS[j] == "OFMAP")
          name_to_output[key] = buffers.size();
        else if (cp.dfe.OUTPUTS[j] == "IFMAP")
          name_to_output[key] = buffers.size() - 1;
        else
          LOG(ERROR) << "Unrecognised output type: " << cp.dfe.OUTPUTS[j]
                     << '\n';
      }

      if (!cp.dfe.RESIDUAL.empty()) {
        auto &residual = buffers[name_to_output[cp.dfe.RESIDUAL]];
        for (int i = 0; i < output_cpu.size(); ++i)
          output_cpu[i] = FixedPointAdd<data_t>(output_cpu[i], residual[i]);
      }

      buffers.push_back(output_cpu);
    }
  }

  auto output_dfe =
      std::vector<data_t>(cps.back().F * cps.back().getOutputHeight() *
                          cps.back().getOutputWidth() * batch_size);

  /* Write to DFE */
  input_dfe = ReorderInput(input_dfe, cps.front(), batch_size);

  constexpr size_t num_bytes_per_burst = 384;
  size_t base_addr = 0;

  BurstAlign(input_dfe,
             num_bytes_per_burst * cps.front().dfe.PC * cps.front().dfe.PK);
  LOG(INFO) << "Tiled input size (burst aligned): " << input_dfe.size();
  BurstAlign(output_dfe,
             num_bytes_per_burst * cps.back().dfe.PF * cps.back().dfe.PK);
  LOG(INFO) << "Tiled output size (burst aligned): " << output_dfe.size();

  base_addr = WriteDRAM<data_t, Dfe>(input_dfe, base_addr, engine);

  /* Launch run */
  Resnet18Wb8_actions_t actions;
  actions.param_batch_size = batch_size;

  LOG(INFO) << "Running ...\n";
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < (int)num_iters; i++) Resnet18Wb8_run(engine, &actions);
  end = std::chrono::system_clock::now();
  LOG(INFO) << "Done\n";

  ReadDRAM<data_t, Dfe>(output_dfe, base_addr, engine);

  output_dfe = ReorderOutput(output_dfe, cps.back(), batch_size);

  for (int i = 0; i < std::min((int)output_dfe.size(), 10); ++i)
    LOG(INFO) << "output_dfe[" << i << "] = "
              << FixedToFloat<data_t>(output_dfe[i],
                                      cps.back().dfe.num_frac_bits)
              << '\n';

  std::chrono::duration<double> elapsed_seconds = end - start;
  LOG(INFO) << "elapsed time: " << elapsed_seconds.count() / (num_iters)
            << "s\n";

  uint64_t ops = 0;
  for (auto &cp : cps) {
    // TODO: check the exact convlayer type.
    uint64_t cur = 0;
    if (cp.dfe.TYPE == "STANDARD")
      cur = (uint64_t)cp.dfe.TOH * cp.dfe.TOW * cp.K * cp.K * cp.C * cp.F * 2;
    else if (cp.dfe.TYPE == "POINTWISE")
      cur = (uint64_t)cp.dfe.TOH * cp.dfe.TOW * cp.C * cp.F * 2;
    else if (cp.dfe.TYPE == "DEPTHWISE_SEPARABLE")
      cur = (uint64_t)cp.dfe.TOH * cp.dfe.TOW * cp.K * cp.K * cp.C * 2 +
            (uint64_t)cp.dfe.TOH * cp.dfe.TOW * cp.C * cp.F * 2;
    LOG(INFO) << "[Ops] layer " << std::setfill(' ') << std::setw(30)
              << cp.dfe.name << " = " << cur * 1e-6 << " M\n";
    ops += cur;
  }
  LOG(INFO) << "OPS:    " << (double)ops * batch_size * 1e-9 << " G\n";
  LOG(INFO) << "FPS:    "
            << (double)batch_size / (elapsed_seconds.count() / num_iters)
            << "\n";
  LOG(INFO) << "GFLOPs: "
            << (double)ops * batch_size * 1e-9 /
                   (elapsed_seconds.count() / num_iters)
            << "\n";

  if (run_golden) {
    auto &output_cpu = buffers.back();
    LOG(INFO) << "Examine results ...\n";
    for (int i = 0; i < 5; ++i)
      LOG(INFO) << "output_cpu[" << i << "] = "
                << FixedToFloat<data_t>(output_cpu[i],
                                        cps.back().dfe.num_frac_bits)
                << '\n';

    for (int i = 0; i < 5; ++i)
      LOG(INFO) << "output_dfe[" << i << "] = "
                << FixedToFloat<data_t>(output_dfe[i],
                                        cps.back().dfe.num_frac_bits)
                << '\n';

    auto output_cpu_float =
        FixedToFloat<data_t>(output_cpu, cps.back().dfe.num_frac_bits);
    auto output_dfe_float =
        FixedToFloat<data_t>(output_dfe, cps.back().dfe.num_frac_bits);

    const int16_t mask = (1 << (cps.back().dfe.num_frac_bits)) - 1;
    LOG(INFO) << "Evaluating difference ...";
    LOG(INFO) << "Mask: " << std::setw(8) << std::showbase << std::setfill('0')
              << std::hex << mask;

    uint64_t num_failed = 0;
    std::vector<double> diffs;
    for (uint64_t i = 0; i < output_cpu.size(); i++) {
      auto diff = std::abs(output_cpu[i] - output_dfe[i]);
      auto float_diff = std::abs(output_cpu_float[i] - output_dfe_float[i]);
      diffs.push_back(float_diff);
      if ((static_cast<int16_t>(diff) | mask) != mask) {
        if (num_failed < 10)
          fprintf(stderr, "Result mis-matched at %6ld: cpu %10.6f dfe %10.6f\n",
                  i, output_cpu_float[i], output_dfe_float[i]);
        num_failed++;
        // exit(1);
      }
    }

    if (output_cpu.size() <= 0)
      LOG(ERROR) << "output_cpu should have positive number of elements\n";

    double total_diff = 0.0f, min_diff = diffs[0], max_diff = diffs[0];
    for (int i = 0; i < diffs.size(); ++i) {
      total_diff += diffs[i];
      min_diff = std::min(min_diff, diffs[i]);
      max_diff = std::max(max_diff, diffs[i]);
    }

    LOG(INFO) << "num_failed: " << num_failed << " total: " << output_cpu.size()
              << " avg_diff: " << (total_diff / output_cpu.size())
              << " min_diff: " << min_diff << " max_diff: " << max_diff << '\n';
  }
  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
