package depthwise_convolution;

import com.maxeler.maxcompiler.v2.build.EngineParameters;

public class DepthwiseConvolutionEngineParameters extends EngineParameters {

  public DepthwiseConvolutionEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    declareParam("TILE_HEIGHT", DataType.INT, 32);
    declareParam("TILE_WIDTH", DataType.INT, 32);
    declareParam("TILE_DEPTH", DataType.INT, 32);
    declareParam("KERNEL_SIZE", DataType.INT, 3);
    declareParam("PAR_WIDTH", DataType.INT, 1);
    declareParam("PAR_DEPTH", DataType.INT, 1);
    declareParam("FREQ", DataType.INT, 200);
    declareParam("USE_WINOGRAD", DataType.BOOL, false);
    declareParam("DEBUG", DataType.BOOL, false);
  }

  public int getTileHeight() {
    return getParam("TILE_HEIGHT");
  }

  public int getTileWidth() {
    return getParam("TILE_WIDTH");
  }

  public int getTileDepth() {
    return getParam("TILE_DEPTH");
  }

  public int getKernelSize() {
    return getParam("KERNEL_SIZE");
  }

  public int getParWidth() {
    return getParam("PAR_WIDTH");
  }

  public int getParDepth() {
    return getParam("PAR_DEPTH");
  }

  public boolean getUseWinograd() {
    return getParam("USE_WINOGRAD");
  }

  public int getFreq() {
    return getParam("FREQ");
  }

  public boolean getDebug() {
    return getParam("DEBUG");
  }

  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_th%d_tw%d_tc%d_pw%d_pc%d_wg%d_%dMHz%s", getMaxFileName(),
        getDFEModel(), getTarget(), getTileHeight(), getTileWidth(), getTileDepth(), getParWidth(),
        getParDepth(), getUseWinograd() ? 1 : 0, getFreq(), (getDebug() ? "_debug" : ""));
  }
}
