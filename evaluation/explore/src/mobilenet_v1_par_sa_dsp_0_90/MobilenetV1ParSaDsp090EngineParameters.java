package mobilenet_v1_par_sa_dsp_0_90;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class MobilenetV1ParSaDsp090EngineParameters extends ConvLayerEngineParameters {

  public MobilenetV1ParSaDsp090EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
