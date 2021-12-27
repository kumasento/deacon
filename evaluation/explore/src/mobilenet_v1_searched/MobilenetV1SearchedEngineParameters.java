package mobilenet_v1_searched;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class MobilenetV1SearchedEngineParameters extends ConvLayerEngineParameters {

  public MobilenetV1SearchedEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
