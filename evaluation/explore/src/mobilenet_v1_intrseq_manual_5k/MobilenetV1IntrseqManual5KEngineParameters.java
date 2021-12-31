package mobilenet_v1_intrseq_manual_5k;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class MobilenetV1IntrseqManual5KEngineParameters extends ConvLayerEngineParameters {

  public MobilenetV1IntrseqManual5KEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
