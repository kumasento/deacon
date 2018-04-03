package depthwise_separable;

import com.maxeler.maxcompiler.v2.build.EngineParameters;

public class DepthwiseSeparableEngineParameters extends EngineParameters {
  public DepthwiseSeparableEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    declareParam("TILE_HEIGHT", DataType.INT, 32);
    declareParam("TILE_WIDTH", DataType.INT, 32);
    declareParam("TILE_IN_DEPTH", DataType.INT, 32);
    declareParam("TILE_OUT_DEPTH", DataType.INT, 32);
    declareParam("KERNEL_SIZE", DataType.INT, 3);
    declareParam("PAR_WIDTH", DataType.INT, 1);
    declareParam("PAR_IN_DEPTH", DataType.INT, 1);
    declareParam("PAR_OUT_DEPTH", DataType.INT, 1);
    declareParam("FREQ", DataType.INT, 200);
    declareParam("DEBUG", DataType.BOOL, false);
  }

  public int getTileHeight() {
    return getParam("TILE_HEIGHT");
  }

  public int getTileWidth() {
    return getParam("TILE_WIDTH");
  }

  public int getTileInDepth() {
    return getParam("TILE_IN_DEPTH");
  }

  public int getTileOutDepth() {
    return getParam("TILE_OUT_DEPTH");
  }

  public int getKernelSize() {
    return getParam("KERNEL_SIZE");
  }

  public int getParWidth() {
    return getParam("PAR_WIDTH");
  }

  public int getParInDepth() {
    return getParam("PAR_IN_DEPTH");
  }

  public int getParOutDepth() {
    return getParam("PAR_OUT_DEPTH");
  }

  public int getFreq() {
    return getParam("FREQ");
  }

  public boolean getDebug() {
    return getParam("DEBUG");
  }

  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_th%d_tw%d_tc%d_tf%d_k%d_pw%d_pc%d_pf%d_%dMHz%s",
        getMaxFileName(), getDFEModel(), getTarget(), getTileHeight(), getTileWidth(),
        getTileInDepth(), getTileOutDepth(), getKernelSize(), getParWidth(), getParInDepth(),
        getParOutDepth(), getFreq(), (getDebug() ? "_debug" : ""));
  }
}
