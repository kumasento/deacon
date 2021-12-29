package mobilenet_v1_par_s16_b16_dsp_0_25;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class MobilenetV1ParS16B16Dsp025EngineParameters extends ConvLayerEngineParameters {

  public MobilenetV1ParS16B16Dsp025EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
