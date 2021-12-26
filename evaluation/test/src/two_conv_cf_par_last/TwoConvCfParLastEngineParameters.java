package two_conv_cf_par_last;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class TwoConvCfParLastEngineParameters extends ConvLayerEngineParameters {

  public TwoConvCfParLastEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
