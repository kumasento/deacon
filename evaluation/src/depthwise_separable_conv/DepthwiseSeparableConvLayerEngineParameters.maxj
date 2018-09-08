package depthwise_separable_conv;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class DepthwiseSeparableConvLayerEngineParameters extends
    ConvLayerEngineParameters {

  private static final String VERSION_NAME = "VERSION";
  private static final int    VERSION      = 1;

  public DepthwiseSeparableConvLayerEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    super.declarations();
    declareParam(VERSION_NAME, DataType.INT, VERSION);
  }

  public int getVersion() {
    return getParam(VERSION_NAME);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_H%d_W%d_C%d_F%d_K%d_f%d_c%d_k%d_SEQ%d_%s_FREQ_%d_V%d",
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
        getFreq(),
        getVersion());
  }
}
