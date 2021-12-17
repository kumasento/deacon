package depthwise_separable_conv;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class DepthwiseSeparableConvLayerEngineParameters extends ConvLayerEngineParameters {
  private static final String VERSION_NAME = "VERSION";
  private static final int VERSION = 1;
  private static final String NUM_LAYER_NAME = "NUM_LAYER";
  private static final int NUM_LAYER = 1;

  public DepthwiseSeparableConvLayerEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    super.declarations();

    declareParam(VERSION_NAME, DataType.INT, VERSION);
    declareParam(NUM_LAYER_NAME, DataType.INT, NUM_LAYER);
  }

  public int getVersion() {
    return getParam(VERSION_NAME);
  }

  public int getNumLayer() {
    return getParam(NUM_LAYER_NAME);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_%d_%d_N%d_H%d_W%d_C%d_F%d_K%d_S%d_f%d_c%d_k%d_SEQ%d_%s_FREQ_%d_V%d%s",
        getMaxFileName(), getDFEModel(), getTarget(), getBitWidth(), getWBW(), getNumFracBits(),
        getNumLayer(), getH(), getW(), getC(), getF(), getK(), getStride(), getPF(), getPC(),
        getPK(), getSeq(), (getUseDRAM() ? "DRAM" : "PCIe"), getFreq(), getVersion(),
        getDebug() ? "_DBG" : "");
  }
}
