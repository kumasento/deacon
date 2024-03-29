package max5_strm_mult;

import com.maxeler.maxcompiler.v2.build.EngineParameters;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.platform.max5.manager.MAX5CManager;

public
class Max5StrmMultManager extends MAX5CManager {

 public
  static final String knlName = "max5_strm_mult_kernel";

 public
  Max5StrmMultManager(EngineParameters params) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    // config.setAllowNonMultipleTransitions(true);
    // config.setDefaultStreamClockFrequency(100);

    // create a kernel block.
    KernelBlock knl = this.addKernel(
        new Max5StrmMultKernel(this.makeKernelParameters(knlName)));

    // connect them to external environment
    for (String inputName : Max5StrmMultKernel.INPUTS)
      knl.getInput(inputName).connect(this.addStreamFromCPU(inputName));
    for (String outputName : Max5StrmMultKernel.OUTPUTS)
      this.addStreamToCPU(outputName).connect(knl.getOutput(outputName));
  }

 public
  EngineInterface interfaceDefault() {
    EngineInterface ei = new EngineInterface();

    // number of ticks
    InterfaceParam N = ei.addParam("N", CPUTypes.INT64);

    // set ticks
    ei.setTicks(knlName, N);

    // set streams
    for (String inputName : Max5StrmMultKernel.INPUTS)
      ei.setStream(inputName, CPUTypes.FLOAT, N * 4);
    for (String outputName : Max5StrmMultKernel.OUTPUTS)
      ei.setStream(outputName, CPUTypes.FLOAT, N * 4);

    return ei;
  }

 public
  static void main(String[] args) {
    EngineParameters params = new EngineParameters(args);

    Max5StrmMultManager mgr = new Max5StrmMultManager(params);
    mgr.createSLiCinterface(mgr.interfaceDefault());

    // BuildConfig cfg = mgr.getBuildConfig();
    // cfg.setBuildEffort(Effort.HIGH);
    // cfg.setMPPRCostTableSearchRange(1, 4);
    // cfg.setMPPRParallelism(4);
    mgr.build();
  }
}
