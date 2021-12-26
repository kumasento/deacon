package bottleneck_shortcut_imbalanced_par;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class BottleneckShortcutImbalancedParEngineParameters extends ConvLayerEngineParameters {

  public BottleneckShortcutImbalancedParEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
