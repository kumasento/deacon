package conv2d;

import com.maxeler.maxcompiler.v2.build.EngineParameters;

public class Conv2DEngineParameters extends EngineParameters {

  private static final String BIT_WIDTH_NAME = "bitWidth";
  private static final int    BIT_WIDTH      = 32;

  private static final String PF_NAME        = "PF";
  private static final int    PF             = 1;
  private static final String PC_NAME        = "PC";
  private static final int    PC             = 1;
  private static final String PK_NAME        = "PK";
  private static final int    PK             = 1;
  private static final String K_NAME         = "K";
  private static final int    K              = 1;
  private static final String VERSION_NAME   = "VERSION";
  private static final int    VERSION        = 1;

  public Conv2DEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    declareParam(BIT_WIDTH_NAME, DataType.INT);
    declareParam(PF_NAME, DataType.INT, PF);
    declareParam(PC_NAME, DataType.INT, PC);
    declareParam(PK_NAME, DataType.INT, PK);
    declareParam(K_NAME, DataType.INT, K);
    declareParam(VERSION_NAME, DataType.INT, VERSION);
  }

  public int getBitWidth() {
    return getParam(BIT_WIDTH_NAME);
  }

  public int getPF() {
    return getParam(PF_NAME);
  }

  public int getPC() {
    return getParam(PC_NAME);
  }

  public int getPK() {
    return getParam(PK_NAME);
  }

  public int getK() {
    return getParam(K_NAME);
  }

  public int getVersion() {
    return getParam(VERSION_NAME);
  }

  @Override
  protected void validate() {
    if (getBitWidth() <= 0)
      throw new IllegalArgumentException("bitWidth should be larger than 0.");
    if (getPF() <= 0)
      throw new IllegalArgumentException("PF should be larger than 0.");
    if (getPC() <= 0)
      throw new IllegalArgumentException("PC should be larger than 0.");
    if (getPK() <= 0)
      throw new IllegalArgumentException("PK should be larger than 0.");
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_K%d_f%d_c%d_k%d",
        getMaxFileName(),
        getDFEModel(),
        getTarget(),
        getBitWidth(),
        getK(),
        getPF(),
        getPC(),
        getPK());
  }
}
