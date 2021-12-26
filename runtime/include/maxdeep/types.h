/// Parameters and other custom types.
#ifndef MAXDEEP_TYPES_H
#define MAXDEEP_TYPES_H

#include <cstdint>
#include <string>

#include "maxdeep/utils.h"

#ifndef NO_DFE
#include "MaxSLiCInterface.h"
#endif

static uint64_t GetConstant(max_file_t *max_file, std::string property) {
  return max_get_constant_uint64t(max_file, property.c_str());
}

static const char *GetStringConstant(max_file_t *max_file,
                                     std::string property) {
  return max_get_constant_string(max_file, property.c_str());
}

struct DfeConvLayerParameters {
  std::vector<uint64_t> PF;
  std::vector<uint64_t> PC;
  uint64_t PK;
  uint64_t TH;
  uint64_t TW;
  uint64_t TOH;
  uint64_t TOW;
  uint64_t TC;
  uint64_t TF;
  uint64_t K;
  uint64_t PAD;
  uint64_t STRIDE;
  std::string TYPE;
  std::string SEQ;
  std::string name;
  std::string RESIDUAL;
  uint64_t NUM_INPUTS, NUM_OUTPUTS;
  std::vector<std::string> INPUTS;
  std::vector<std::string> OUTPUTS;
  bool wino_coeff_offline;
  bool coeff_on_chip;
  uint64_t num_frac_bits;

  DfeConvLayerParameters() {}
  DfeConvLayerParameters(uint64_t PF, uint64_t PC, uint64_t PK,
                         int num_frac_bits)
      : PF(PF), PC(PC), PK(PK), num_frac_bits(num_frac_bits) {}

#ifndef NO_DFE
  static DfeConvLayerParameters get(max_file_t *max_file, std::string name) {
    DfeConvLayerParameters dcp;

    dcp.name = name;
    dcp.TH = GetConstant(max_file, name + "_H");
    dcp.TW = GetConstant(max_file, name + "_W");
    dcp.TOH = GetConstant(max_file, name + "_OH");
    dcp.TOW = GetConstant(max_file, name + "_OW");
    dcp.TC = GetConstant(max_file, name + "_C");
    dcp.TF = GetConstant(max_file, name + "_F");
    dcp.PK = GetConstant(max_file, name + "_PK");
    dcp.PAD = GetConstant(max_file, name + "_PAD");
    dcp.STRIDE = GetConstant(max_file, name + "_STRIDE");
    dcp.K = GetConstant(max_file, name + "_K");
    dcp.TYPE = GetStringConstant(max_file, name + "_TYPE");
    dcp.SEQ = GetStringConstant(max_file, name + "_SEQ");
    dcp.coeff_on_chip = GetConstant(max_file, name + "_COEFF_ON_CHIP") == 1;
    dcp.num_frac_bits = GetConstant(max_file, name + "_num_frac_bits");
    dcp.RESIDUAL = GetStringConstant(max_file, name + "_RESIDUAL");
    dcp.NUM_INPUTS = GetConstant(max_file, name + "_NUM_INPUTS");
    dcp.NUM_OUTPUTS = GetConstant(max_file, name + "_NUM_OUTPUTS");
    for (uint64_t i = 0; i < dcp.NUM_INPUTS; ++i)
      dcp.INPUTS.push_back(
          GetStringConstant(max_file, name + "_INPUT_" + std::to_string(i)));
    for (uint64_t i = 0; i < dcp.NUM_OUTPUTS; ++i)
      dcp.OUTPUTS.push_back(
          GetStringConstant(max_file, name + "_OUTPUT_" + std::to_string(i)));

    for (uint64_t i = 0; i < dcp.NUM_INPUTS; ++i)
      dcp.PC.push_back(
          GetConstant(max_file, name + "_PC_" + std::to_string(i)));
    for (uint64_t i = 0; i < dcp.NUM_OUTPUTS; ++i)
      dcp.PF.push_back(
          GetConstant(max_file, name + "_PF_" + std::to_string(i)));
    // Globals
    dcp.wino_coeff_offline = GetConstant(max_file, "WINO_COEFF_OFFLINE") == 1;

    return dcp;
  }
#endif

  void dump() {
    // LOG(INFO) << "\n + DFE parameters:"
    //           << "\n   * PC = " << PC << "\n   * PF = " << PF
    //           << "\n   * PK = " << PK << '\n';
  }
};

struct ConvLayerParameters {
  int H, W, C, F, K, P, S;
  DfeConvLayerParameters dfe;

  ConvLayerParameters(max_file_t *max_file, std::string name)
      : dfe{DfeConvLayerParameters::get(max_file, name)} {
    // Initialise other parameters from the design spec.
    // H = GetConvLayerInputDim(dfe.TH, dfe.K, 0, 1);
    // W = GetConvLayerInputDim(dfe.TW, dfe.K, 0, 1);
    H = dfe.TH;
    W = dfe.TW;
    C = dfe.TC;
    F = dfe.TF;
    K = dfe.K;
    P = dfe.PAD;  // TODO: change these
    S = dfe.STRIDE;
  }
  ConvLayerParameters(int C, int F, int K, int P, int S, uint64_t PF,
                      uint64_t PC, uint64_t PK, int num_frac_bits)
      : C(C), F(F), K(K), dfe{PF, PC, PK, num_frac_bits} {}

  int getOutputHeight() const { return GetConvLayerOutputDim(H, K, P, S); }
  int getOutputWidth() const { return GetConvLayerOutputDim(W, K, P, S); }

  void dump() {
    LOG(INFO) << "ConvLayerParameters:"
              << "\n + H = " << H << "\n + W = " << W << "\n + K = " << K
              << "\n + P = " << P << "\n + S = " << S << '\n';

    dfe.dump();
  }
};

#endif
