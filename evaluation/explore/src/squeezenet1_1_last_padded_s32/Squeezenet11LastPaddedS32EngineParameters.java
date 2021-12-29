package squeezenet1_1_last_padded_s32;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class Squeezenet11LastPaddedS32EngineParameters extends ConvLayerEngineParameters {

  public Squeezenet11LastPaddedS32EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
