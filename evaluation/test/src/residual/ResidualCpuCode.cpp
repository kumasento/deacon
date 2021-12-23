#include <getopt.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "Maxfiles.h"
#include "maxdeep/layers.h"
#include "maxdeep/utils.h"



typedef int16_t data_t;


struct Dfe {
  typedef Residual_actions_t dfe_run_actions_t;
  typedef Residual_dramRead_actions_t dram_read_actions_t;
  typedef Residual_dramWrite_actions_t dram_write_actions_t;

  static void ReadDRAM(max_engine_t *engine, dram_read_actions_t *actions) {
    Residual_dramRead_run(engine, actions);
  }
  static void WriteDRAM(max_engine_t *engine, dram_write_actions_t *actions) {
    Residual_dramWrite_run(engine, actions);
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

  max_file_t *max_file = Residual_init();
  max_engine_t *engine = max_load(max_file, "*");

  std::vector<ConvLayerParameters> cps;
	cps.push_back(ConvLayerParameters(max_file, "conv0"));
	cps.push_back(ConvLayerParameters(max_file, "conv1"));


  /* Generate input data */
  float max_val = 1.1, min_val = -1.1, input_scale = 1;
  auto input =
      CreateRandomArray<float>(cps.front().dfe.TC * cps.front().dfe.TH * cps.front().dfe.TW * batch_size,
                               input_scale * min_val, input_scale * max_val);
                        LOG(INFO) << "num frac bits: " << cps.front().dfe.num_frac_bits << '\n';
  auto input_dfe = FloatToFixed<data_t>(input, cps.front().dfe.num_frac_bits);

  for (int i = 0; i < 3; ++ i)
    LOG(INFO) << "input[" << i << "] = " << input[i] << "  dfe = " <<  FixedToFloat<data_t>(input_dfe[i], cps.front().dfe.num_frac_bits) << '\n';

  auto output_dfe =
      std::vector<data_t>(cps.back().F * cps.back().getOutputHeight() *
                          cps.back().getOutputWidth() * batch_size);

  /* Write to DFE */
  input_dfe = ReorderInput(input_dfe, cps.front(), batch_size);

  constexpr size_t num_bytes_per_burst = 384;
  size_t base_addr = 0;

  BurstAlign(input_dfe, num_bytes_per_burst * cps.front().dfe.PC * cps.front().dfe.PK);
  LOG(INFO) << "Tiled input size (burst aligned): " << input_dfe.size();
  BurstAlign(output_dfe, num_bytes_per_burst * cps.back().dfe.PF * cps.back().dfe.PK);
  LOG(INFO) << "Tiled output size (burst aligned): " << output_dfe.size();

  base_addr = WriteDRAM<data_t, Dfe>(input_dfe, base_addr, engine);

  /* Launch run */
  Residual_actions_t actions;
  actions.param_batch_size = batch_size;


  LOG(INFO) << "Running ...\n";
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  for (int i = 0; i < (int)num_iters; i++)
    Residual_run(engine, &actions);
  end = std::chrono::system_clock::now();
  LOG(INFO) << "Done\n";

  ReadDRAM<data_t, Dfe>(output_dfe, base_addr, engine);

  output_dfe = ReorderOutput(output_dfe, cps.back(), batch_size);

  for (int i = 0; i < std::min((int)output_dfe.size(), 10); ++ i) 
    LOG(INFO) << "output_dfe[" << i << "] = " << FixedToFloat<data_t>(output_dfe[i], cps.back().dfe.num_frac_bits) << '\n';
  

  std::chrono::duration<double> elapsed_seconds = end - start;
  LOG(INFO) << "elapsed time: " << elapsed_seconds.count() / (num_iters) << "s\n";

  uint64_t ops = 0;
  for (auto &cp : cps) {
    // TODO: check the exact convlayer type.
    if (cp.dfe.TYPE == "STANDARD")
      ops += cp.H * cp.W * cp.K * cp.K * cp.C * cp.F * 2;
    else if (cp.dfe.TYPE == "POINTWISE")
      ops += cp.H * cp.W * cp.C * cp.F * 2;
    else if (cp.dfe.TYPE == "DEPTHWISE_SEPARABLE")
      ops += cp.H * cp.W * cp.K * cp.K * cp.C * 2 + cp.H * cp.W * cp.C * cp.F * 2;
  }
  LOG(INFO) << "FPS:    " << (double) batch_size / (elapsed_seconds.count() / num_iters) << "\n";
  LOG(INFO) << "GFLOPs: " << (double)ops * batch_size * 1e-9 / (elapsed_seconds.count() / num_iters) << "\n";

  max_unload(engine);
  max_file_free(max_file);

  return 0;
}
