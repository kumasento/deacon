package dotprod;

import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFETypeFactory;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.platform.max5.manager.Max5LimaManager;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.BuildConfig.Effort;
import com.maxeler.platform.max5.manager.BuildConfig.OptimizationGoal;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceMath;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

public
final class DotProdMax5LimaManager extends Max5LimaManager {

 private
  final DFEType dfeT = DFETypeFactory.dfeFloat(8, 24);
 private
  final CPUTypes cpuT = CPUTypes.FLOAT;
 private
  final String KERNEL_NAME = "DOT_PROD_KERNEL";
 private
  final int vecSize;

 public
  DotProdMax5LimaManager(DotProdEngineParameters params) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(200);

    this.vecSize = params.getVecSize();

    KernelBlock blk = addKernel(
        new DotProdKernel(makeKernelParameters(KERNEL_NAME),
                          params.getVecSize(), params.getBitWidth(), dfeT));

    blk.getInput(DotProdKernel.VEC_A).connect(addStreamFromCPU("VEC_A"));
    blk.getInput(DotProdKernel.VEC_B).connect(addStreamFromCPU("VEC_B"));
    addStreamToCPU("RESULT").connect(blk.getOutput(DotProdKernel.RES));

    addMaxFileConstant("VEC_LEN", vecSize);
  }

 public
  EngineInterface interfaceDefault() {
    EngineInterface ei = new EngineInterface();

    InterfaceParam M =
        ei.addParam("M", CPUTypes.INT64);  // total number of vectors
    InterfaceParam N = ei.addParam("N", CPUTypes.INT64);  // vector length
    InterfaceParam L = ei.getAutoLoopOffset(KERNEL_NAME, DotProdKernel.OFFSET);
    InterfaceParam numVecs =
        InterfaceMath
            .ceil(N.cast(CPUTypes.DOUBLE) / ei.addConstant((double)vecSize))
            .cast(CPUTypes.UINT32);
    ei.setTicks(KERNEL_NAME, M * numVecs * L);
    ei.setStream("VEC_A", cpuT, cpuT.sizeInBytes() * M * N);
    ei.setStream("VEC_B", cpuT, cpuT.sizeInBytes() * M * N);
    ei.setStream("RESULT", cpuT, M * cpuT.sizeInBytes());
    ei.setScalar(KERNEL_NAME, DotProdKernel.NUM_VECS, numVecs);
    ei.ignoreAutoLoopOffset(KERNEL_NAME, DotProdKernel.OFFSET);

    return ei;
  }

  /**
   * @param args
   */
 public
  static void main(String[] args) {
    DotProdEngineParameters params = new DotProdEngineParameters(args);
    DotProdMax5LimaManager mgr = new DotProdMax5LimaManager(params);
    mgr.createSLiCinterface(mgr.interfaceDefault());

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    // buildConfig.setMPPRCostTableSearchRange(1, 4);
    // buildConfig.setMPPRParallelism(4);

    mgr.build();
  }
}