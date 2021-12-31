package single_conv_b8_p2;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class SingleConvB8P2EngineParameters extends ConvLayerEngineParameters {

  public SingleConvB8P2EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
