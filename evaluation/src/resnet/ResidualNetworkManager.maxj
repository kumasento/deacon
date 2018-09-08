package resnet;

import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;

public class ResidualNetworkManager extends CustomManager {
  public ResidualNetworkManager(ResidualNetworkEngineParameters params) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);
    config.setDefaultStreamClockFrequency(params.getFreq());


  }

  public EngineInterface interfaceDefault() {
    EngineInterface ei = new EngineInterface();

    return ei;
  }

  public static void main(String[] args) {
    ResidualNetworkEngineParameters params = new ResidualNetworkEngineParameters(args);

    ResidualNetworkManager mgr = new ResidualNetworkManager(params);
    mgr.createSLiCinterface(mgr.interfaceDefault());

    BuildConfig cfg = mgr.getBuildConfig();
    cfg.setBuildEffort(Effort.HIGH);
    cfg.setMPPRCostTableSearchRange(1, 4);
    cfg.setMPPRParallelism(4);
    mgr.build();
  }

}
