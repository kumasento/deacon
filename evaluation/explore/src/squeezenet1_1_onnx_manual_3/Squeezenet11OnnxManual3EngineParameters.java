package squeezenet1_1_onnx_manual_3;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class Squeezenet11OnnxManual3EngineParameters extends ConvLayerEngineParameters {

  public Squeezenet11OnnxManual3EngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
