package bottleneck_shortcut_imbalanced;

import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;

public class BottleneckShortcutImbalancedEngineParameters extends ConvLayerEngineParameters {

  public BottleneckShortcutImbalancedEngineParameters(String[] args) {
    super(args);
  }
  
  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_FREQ_%d%s", getMaxFileName(), getDFEModel(), getTarget(), getFreq(), getDebug() ? "_DBG" : ""); 
  }
}
