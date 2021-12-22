package squeezenet_fire;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class SqueezenetFireEngineParameters extends ConvLayerEngineParameters {

  public SqueezenetFireEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
