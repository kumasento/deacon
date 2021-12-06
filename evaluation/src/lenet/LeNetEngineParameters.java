package lenet;

import java.util.ArrayList;
import java.util.List;

import com.maxeler.maxcompiler.v2.build.EngineParameters;

public class LeNetEngineParameters extends EngineParameters {

  private static final String  BW_NAME        = "BW";
  private static final int     BW             = 32;
  private static final String  PP_NAME        = "PP";
  private static final String  PP             = "1,1,1";
  private static final String  FREQ_NAME      = "FREQ";
  private static final int     FREQ           = 100;
  private static final String  USE_DEPTH_NAME = "USE_DEPTH";
  private static final boolean USE_DEPTH      = false;

  public LeNetEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    declareParam(BW_NAME, DataType.INT, BW);
    declareParam(PP_NAME, DataType.STRING, PP);
    declareParam(FREQ_NAME, DataType.INT, FREQ);
    declareParam(USE_DEPTH_NAME, DataType.BOOL, USE_DEPTH);
  }

  public int getBW() {
    return getParam(BW_NAME);
  }

  public List<Integer> getPP() {
    List<Integer> pp = new ArrayList<Integer>();
    String rawPP = getParam(PP_NAME);
    for (String p : rawPP.split(",")) {
      pp.add(Integer.parseInt(p));
    }

    return pp;
  }

  public int getFreq() {
    return getParam(FREQ_NAME);
  }

  public boolean getUseDepth() {
    return getParam(USE_DEPTH_NAME);
  }

  @Override
  public String getBuildName() {
    return String.format(
        "%s_%s_%s_b%d_PP%s_%s_FREQ_%d",
        getMaxFileName(),
        getDFEModel(),
        getTarget(),
        getBW(),
        ((String) getParam(PP_NAME)).replaceAll(",", "_"),
        (!getUseDepth()) ? "STD" : "DWS",
        getFreq());
  }
}