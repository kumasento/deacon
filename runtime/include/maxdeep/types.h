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

struct DfeConvLayerParameters {
  uint64_t PF;
  uint64_t PC;
  uint64_t PK;
  uint64_t TH;
  uint64_t TW;
  uint64_t TC;
  uint64_t TF;
  uint64_t K;
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

    dcp.TH = GetConstant(max_file, name + "_H");
    dcp.TW = GetConstant(max_file, name + "_W");
    dcp.TC = GetConstant(max_file, name + "_C");
    dcp.TF = GetConstant(max_file, name + "_F");
    dcp.PC = GetConstant(max_file, name + "_PC");
    dcp.PF = GetConstant(max_file, name + "_PF");
    dcp.PK = GetConstant(max_file, name + "_PK");
    dcp.K = GetConstant(max_file, name + "_K");
    dcp.coeff_on_chip = GetConstant(max_file, name + "_COEFF_ON_CHIP") == 1;
    dcp.num_frac_bits = GetConstant(max_file, name + "_num_frac_bits");

    // Globals
    dcp.wino_coeff_offline = GetConstant(max_file, "WINO_COEFF_OFFLINE") == 1;

    return dcp;
  }
#endif
};

struct ConvLayerParameters {
  int H, W, C, F, K, P, S;
  DfeConvLayerParameters dfe;

  ConvLayerParameters(max_file_t *max_file, std::string name)
      : dfe{DfeConvLayerParameters::get(max_file, name)} {
    // Initialise other parameters from the design spec.
    H = dfe.TH;
    W = dfe.TW;
    C = dfe.TC;
    F = dfe.TF;
    K = dfe.K;
    P = 0;  // TODO: change these
    S = 1;
  }
  ConvLayerParameters(int C, int F, int K, int P, int S, uint64_t PF,
                      uint64_t PC, uint64_t PK, int num_frac_bits)
      : C(C), F(F), K(K), dfe{PF, PC, PK, num_frac_bits} {}

  int getOutputHeight() const { return GetConvLayerOutputDim(H, K, P, S); }
  int getOutputWidth() const { return GetConvLayerOutputDim(W, K, P, S); }
};

#endif