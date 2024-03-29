package com.custom_computing_ic.maxdeep.manager.conv_single_layer;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class ConvSingleLayerEngineParameters extends ConvLayerEngineParameters {
  public ConvSingleLayerEngineParameters(String[] args) {
    super(args);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_H%d_W%d_C%d_F%d_K%d_f%d_c%d_k%d_SEQ%d_%s_FREQ_%d%s%s%s%s%s",
        getMaxFileName(), getDFEModel(), getTarget(), getBitWidth(), getH(),
        getW(), getC(), getF(), getK(), getPF(), getPC(), getPK(), getSeq(),
        (getUseDRAM() ? "DRAM" : "PCIe"), getFreq(),
        (getUseWinograd() ? "_WINO" : ""),
        (getWinogradWeightsOffline() ? "_COEF" : ""),
        (getNumCoeffFifoSplits() > 1 ? "_S" + getNumCoeffFifoSplits() : ""),
        (getCoeffOnChip() ? "_COC" : ""),
        (getDebug() ? "_DEBUG" : ""));
  }
}
