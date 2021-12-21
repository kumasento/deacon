package three_conv_fff;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class ThreeConvFffEngineParameters extends ConvLayerEngineParameters {

  public ThreeConvFffEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
