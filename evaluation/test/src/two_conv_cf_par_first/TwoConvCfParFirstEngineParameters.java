package two_conv_cf_par_first;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class TwoConvCfParFirstEngineParameters extends ConvLayerEngineParameters {

  public TwoConvCfParFirstEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
