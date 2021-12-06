package dotprod;

import com.maxeler.maxcompiler.v2.build.EngineParameters;

public class DotProdEngineParameters extends EngineParameters {

  private static final String BIT_WIDTH_NAME = "bitWidth";
  private static final int BIT_WIDTH = 32;
  
  private static final String VEC_SIZE_NAME = "vecSize";
  private static final int VEC_SIZE = 9;

  public DotProdEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    declareParam(BIT_WIDTH_NAME, DataType.INT);
    declareParam(VEC_SIZE_NAME, DataType.INT);
  }

  public int getBitWidth() {
    return getParam(BIT_WIDTH_NAME);
  }
  
  public int getVecSize() {
    return getParam(VEC_SIZE_NAME);
  }

  @Override
  protected void validate() {
    if (getBitWidth() <= 0)
      throw new IllegalArgumentException("bitWidth should be larger than 0.");
    if (getVecSize() <= 0)
      throw new IllegalArgumentException("vecSize should be larger than 0.");
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_n%d",
        getMaxFileName(),
        getDFEModel(),
        getTarget(),
        getBitWidth(),
        getVecSize());
  }
}
