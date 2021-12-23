package squeezenet_two_fires;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class SqueezenetTwoFiresEngineParameters extends ConvLayerEngineParameters {

  public SqueezenetTwoFiresEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
