package single_dws_conv_non_divisible;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class SingleDwsConvNonDivisibleEngineParameters extends ConvLayerEngineParameters {

  public SingleDwsConvNonDivisibleEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
