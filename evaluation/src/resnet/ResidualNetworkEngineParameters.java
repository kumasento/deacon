package resnet;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class ResidualNetworkEngineParameters extends ConvLayerEngineParameters {
  private static final String FS_NAME = "FS"; // filters
  private static final String FS = "64,64,256";
  private static final String PFS_NAME = "PFS"; // filters
  private static final String PFS = "1,1,1";
  private static final String PH_NAME = "PH";
  private static final int PH = 1;
  private static final String PW_NAME = "PW";
  private static final int PW = 1;

  public ResidualNetworkEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    super.declarations();

    declareParam(FS_NAME, DataType.STRING, FS);
    declareParam(PFS_NAME, DataType.STRING, PFS);
    declareParam(PH_NAME, DataType.INT, PH);
    declareParam(PW_NAME, DataType.INT, PW);
  }

  public int getF(int idx) {
    String[] filters = ((String) getParam(FS_NAME)).split(",");
    return Integer.parseInt(filters[idx]);
  }

  public int getPF(int idx) {
    String[] parFilters = ((String) getParam(PFS_NAME)).split(",");
    return Integer.parseInt(parFilters[idx]);
  }

  public int getPH() {
    return getParam(PH_NAME);
  }

  public int getPW() {
    return getParam(PW_NAME);
  }

  @Override
  public String getBuildName() {
    String buildName = String.format("%s_%s_%s", getMaxFileName(), getDFEModel(), getTarget());

    buildName += String.format("_FREQ_%d", getFreq());
    buildName += String.format("_b%d", getBitWidth());

    buildName += String.format("_H%d_W%d_C%d", getH(), getW(), getC());
    buildName += String.format("_F%d_%d_%d", getF(0), getF(1), getF(2));
    buildName += String.format("_PH%d_PW%d_PC%d", getPH(), getPW(), getPC());
    buildName += String.format("_PF%d_%d_%d", getPF(0), getPF(1), getPF(2));

    buildName += "_" + (getUseDRAM() ? "DRAM" : "PCIe");

    if (getUseWinograd()) // use winograd or not
      buildName += "_WINO";
    if (getNumCoeffFifoSplits() > 1) // coeffient FIFO splits
      buildName += "_S" + getNumCoeffFifoSplits();
    if (getDebug()) // whether to add debug flag
      buildName += "_DEBUG";

    return buildName;
  }
}
