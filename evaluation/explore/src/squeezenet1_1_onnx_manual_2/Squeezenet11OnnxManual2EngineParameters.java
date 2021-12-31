package squeezenet1_1_onnx_manual_2;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class Squeezenet11OnnxManual2EngineParameters extends ConvLayerEngineParameters {

  public Squeezenet11OnnxManual2EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
