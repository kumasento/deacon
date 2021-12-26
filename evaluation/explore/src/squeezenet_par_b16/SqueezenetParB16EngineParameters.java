package squeezenet_par_b16;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class SqueezenetParB16EngineParameters extends ConvLayerEngineParameters {

  public SqueezenetParB16EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
