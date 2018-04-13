package shift_mult_compare;

import com.maxeler.maxcompiler.v2.build.EngineParameters;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

public class ShiftMultCompareManager extends CustomManager {

  public static final String knlName = "shift_mult_compare_kernel";

  public ShiftMultCompareManager(EngineParameters params) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);
    config.setDefaultStreamClockFrequency(100);

    // create a kernel block.
    KernelBlock knl =
        this.addKernel(new ShiftMultCompareKernel(this.makeKernelParameters(knlName)));

    // connect them to external environment
    for (String inputName : ShiftMultCompareKernel.INPUTS)
      knl.getInput(inputName).connect(this.addStreamFromCPU(inputName));
    for (String outputName : ShiftMultCompareKernel.OUTPUTS)
      this.addStreamToCPU(outputName).connect(knl.getOutput(outputName));
  }

  public EngineInterface interfaceDefault() {
    EngineInterface ei = new EngineInterface();

    // number of ticks
    InterfaceParam N = ei.addParam("N", CPUTypes.INT64);

    // set ticks
    ei.setTicks(knlName, N);

    // set streams
    for (String inputName : ShiftMultCompareKernel.INPUTS)
      ei.setStream(inputName, CPUTypes.UINT16, N);
    for (String outputName : ShiftMultCompareKernel.OUTPUTS)
      ei.setStream(outputName, CPUTypes.UINT16, N);

    return ei;
  }

  public static void main(String[] args) {
    EngineParameters params = new EngineParameters(args);

    ShiftMultCompareManager mgr = new ShiftMultCompareManager(params);
    mgr.createSLiCinterface(mgr.interfaceDefault());

    BuildConfig cfg = mgr.getBuildConfig();
    cfg.setBuildEffort(Effort.HIGH);
    cfg.setMPPRCostTableSearchRange(1, 4);
    cfg.setMPPRParallelism(4);
    mgr.build();
  }

}
