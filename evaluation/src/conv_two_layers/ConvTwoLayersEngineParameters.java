package conv_two_layers;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class ConvTwoLayersEngineParameters extends ConvLayerEngineParameters {
  private static final String SEQ0_NAME = "SEQ0";
  private static final int SEQ0 = 0;
  private static final String SEQ1_NAME = "SEQ1";
  private static final int SEQ1 = 0;

  public ConvTwoLayersEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    super.declarations();

    declareParam(SEQ0_NAME, DataType.INT, SEQ0);
    declareParam(SEQ1_NAME, DataType.INT, SEQ1);
  }

  public int getSeq0() {
    return getParam(SEQ0_NAME);
  }

  public int getSeq1() {
    return getParam(SEQ1_NAME);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_H%d_W%d_C%d_F%d_K%d_f%d_c%d_k%d_SEQ%d_%d_%s%s_FREQ_%d",
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
        getSeq0(),
        getSeq1(),
        (getUseDRAM() ? "DRAM" : "PCIe"),
        (getCoeffOnChip() ? "_COC" : ""),
        getFreq());
  }
}
