package bottleneck;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class BottleneckEngineParameters extends ConvLayerEngineParameters {
  private static final String SEQ0_NAME = "SEQ0";
  private static final int SEQ0 = 0;
  private static final String SEQ1_NAME = "SEQ1";
  private static final int SEQ1 = 0;
  private static final String SEQ2_NAME = "SEQ2";
  private static final int SEQ2 = 0;

  public BottleneckEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    super.declarations();

    declareParam(SEQ0_NAME, DataType.INT, SEQ0);
    declareParam(SEQ1_NAME, DataType.INT, SEQ1);
    declareParam(SEQ2_NAME, DataType.INT, SEQ2);
  }

  public int getSeq0() {
    return getParam(SEQ0_NAME);
  }

  public int getSeq1() {
    return getParam(SEQ1_NAME);
  }

  public int getSeq2() {
    return getParam(SEQ2_NAME);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_%d_H%d_W%d_C%d_F%d_K%d_S%d_f%d_c%d_k%d_SEQ%d_%d_%d_%s%s_FREQ_%d%s",
        getMaxFileName(),
        getDFEModel(),
        getTarget(),
        getBitWidth(),
        getWBW(),
        getH(),
        getW(),
        getC(),
        getF(),
        getK(),
        getStride(),
        getPF(),
        getPC(),
        getPK(),
        getSeq0(),
        getSeq1(),
        getSeq2(),
        (getUseDRAM() ? "DRAM" : "PCIe"),
        (getCoeffOnChip() ? "_COC" : ""),
        getFreq(),
        (getDebug() ? "_DBG" : ""));
  }
}
