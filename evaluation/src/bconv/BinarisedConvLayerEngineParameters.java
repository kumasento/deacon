package bconv;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class BinarisedConvLayerEngineParameters extends
    ConvLayerEngineParameters {

  public BinarisedConvLayerEngineParameters(String[] args) {
    super(args);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_H%d_W%d_C%d_F%d_K%d_f%d_c%d_k%d_SEQ%d_%s_FREQ_%d",
        getMaxFileName(),
        getDFEModel(),
        getTarget(),
        getBitWidth(),
        getH(),
        getW(),
        getC(),
        getF(),
        getK(),
        getPF(),
        getPC(),
        getPK(),
        getSeq(),
        (getUseDRAM() ? "DRAM" : "PCIe"),
        getFreq());
  }
}
