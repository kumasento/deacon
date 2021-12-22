package resnet_18_fst_id;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class Resnet18FstIdEngineParameters extends ConvLayerEngineParameters {

  public Resnet18FstIdEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
