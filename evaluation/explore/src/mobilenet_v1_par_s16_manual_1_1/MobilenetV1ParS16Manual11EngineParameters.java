package mobilenet_v1_par_s16_manual_1_1;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class MobilenetV1ParS16Manual11EngineParameters extends ConvLayerEngineParameters {

  public MobilenetV1ParS16Manual11EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
