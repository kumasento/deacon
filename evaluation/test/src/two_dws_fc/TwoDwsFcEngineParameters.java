package two_dws_fc;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class TwoDwsFcEngineParameters extends ConvLayerEngineParameters {

  public TwoDwsFcEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
