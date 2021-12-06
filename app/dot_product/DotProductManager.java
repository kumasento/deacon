package dot_product;

import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFETypeFactory;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceMath;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.platform.max5.manager.MAX5CManager;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.build.EngineParameters;

public final class DotProductManager extends MAX5CManager {

  public DotProductManager(EngineParameters params) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    setAllowNonMultipleTransitions(true);
    setDefaultStreamClockFrequency(200);

    KernelBlock blk = addKernel(new DotProductKernel(makeKernelParameters("DotProduct")));

    blk.getInput("x").connect(addStreamFromCPU("x"));
    blk.getInput("y").connect(addStreamFromCPU("y"));
    addStreamToCPU("z").connect(blk.getOutput("z"));
  }

  public EngineInterface interfaceDefault() {
    EngineInterface ei = new EngineInterface();

    InterfaceParam N = ei.addParam("N", CPUTypes.INT64); // stream size.

    ei.setTicks("DotProduct", N);
    ei.setStream("x", CPUTypes.FLOAT, N.mul(CPUTypes.FLOAT.sizeInBytes()));
    ei.setStream("y", CPUTypes.FLOAT, N.mul(CPUTypes.FLOAT.sizeInBytes()));
    ei.setStream("z", CPUTypes.FLOAT, N.mul(CPUTypes.FLOAT.sizeInBytes()));

    return ei;
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    EngineParameters params = new EngineParameters(args);
    DotProductManager mgr = new DotProductManager(params);
    mgr.createSlicInterface(mgr.interfaceDefault());

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);

    mgr.build();
  }
}
