package conv_stride;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class ConvStrideEngineParameters extends ConvLayerEngineParameters {
  public ConvStrideEngineParameters(String[] args) {
    super(args);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_H%d_W%d_C%d_F%d_K%d_PAD%d_S%d_f%d_c%d_k%d_SEQ%d_%s%s_FREQ_%d%s",
        getMaxFileName(),
        getDFEModel(),
        getTarget(),
        getBitWidth(),
        getH(),
        getW(),
        getC(),
        getF(),
        getK(),
        getPad(),
        getStride(),
        getPF(),
        getPC(),
        getPK(),
        getSeq(),
        (getUseDRAM() ? "DRAM" : "PCIe"),
        (getCoeffOnChip() ? "_COC" : ""),
        getFreq(),
        (getDebug() ? "_DBG" : ""));
  }
}
