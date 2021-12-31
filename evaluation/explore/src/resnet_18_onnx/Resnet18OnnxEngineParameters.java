package resnet_18_onnx;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class Resnet18OnnxEngineParameters extends ConvLayerEngineParameters {

  public Resnet18OnnxEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
