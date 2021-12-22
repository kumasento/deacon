package resnet_18_fst_maxpool;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class Resnet18FstMaxpoolEngineParameters extends ConvLayerEngineParameters {

  public Resnet18FstMaxpoolEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
