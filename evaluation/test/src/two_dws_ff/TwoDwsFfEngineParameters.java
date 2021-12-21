package two_dws_ff;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class TwoDwsFfEngineParameters extends ConvLayerEngineParameters {

  public TwoDwsFfEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
