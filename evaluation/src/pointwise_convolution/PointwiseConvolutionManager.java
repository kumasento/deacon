package pointwise_convolution;

import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.custom_computing_ic.maxdeep.manager.Max5LMemManager;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFETypeFactory;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.BuildConfig.Effort;
import com.maxeler.platform.max5.manager.BuildConfig.OptimizationGoal;

public
class PointwiseConvolutionManager extends Max5LMemManager implements
    ManagerInterface {

 public
  static final String knlName = "pointwise_convolution_kernel";
 private
  static DFEType dfeT;
 private
  static CPUTypes cpuT;

 public
  PointwiseConvolutionManager(PointwiseConvolutionEngineParameters params) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

    dfeT = DFETypeFactory.dfeFloat(8, 24);
    cpuT = CPUTypes.FLOAT;

    // constants
    addMaxFileConstant("TILE_HEIGHT", params.getTileHeight());
    addMaxFileConstant("TILE_WIDTH", params.getTileWidth());
    addMaxFileConstant("TILE_IN_DEPTH", params.getTileInDepth());
    addMaxFileConstant("TILE_OUT_DEPTH", params.getTileOutDepth());
    addMaxFileConstant("PAR_WIDTH", params.getParWidth());
    addMaxFileConstant("PAR_IN_DEPTH", params.getParInDepth());
    addMaxFileConstant("PAR_OUT_DEPTH", params.getParOutDepth());

    // create a kernel block.
    KernelBlock knl = this.addKernel(new PointwiseConvolutionKernel(
        this.makeKernelParameters(knlName), params.getTileHeight(),
        params.getTileWidth(), params.getTileInDepth(),
        params.getTileOutDepth(), params.getParWidth(), params.getParInDepth(),
        params.getParOutDepth(), dfeT, params.getDebug()));

    // connect them to external environment
    for (String inputName : PointwiseConvolutionKernel.INPUTS)
      knl.getInput(inputName).connect(this.addStreamFromCPU(inputName));
    for (String outputName : PointwiseConvolutionKernel.OUTPUTS)
      this.addStreamToCPU(outputName).connect(knl.getOutput(outputName));
  }

 public
  EngineInterface interfaceDefault(
      PointwiseConvolutionEngineParameters params) {
    EngineInterface ei = new EngineInterface();

    // number of computation batches
    InterfaceParam N = ei.addParam("N", CPUTypes.INT64);
    InterfaceParam numParWidth =
        ei.addConstant(params.getTileWidth() / params.getParWidth());
    InterfaceParam numParInDepth =
        ei.addConstant(params.getTileInDepth() / params.getParInDepth());
    InterfaceParam numParOutDepth =
        ei.addConstant(params.getTileOutDepth() / params.getParOutDepth());
    InterfaceParam numTicks = params.getTileHeight() * numParWidth *
                              numParInDepth * numParOutDepth * N;

    InterfaceParam numIfmapElems =
        ei.addConstant(params.getTileHeight() * params.getTileWidth() *
                       params.getTileInDepth());
    InterfaceParam numOfmapElems =
        ei.addConstant(params.getTileHeight() * params.getTileWidth() *
                       params.getTileOutDepth());
    InterfaceParam numWeightsElems =
        ei.addConstant(params.getTileOutDepth() * params.getTileInDepth());
    InterfaceParam numBiasElems = ei.addConstant(params.getTileOutDepth());

    // set ticks
    ei.setTicks(knlName, numTicks);
    ei.setStream(PointwiseConvolutionKernel.INPUTS[0], cpuT,
                 cpuT.sizeInBytes() * numIfmapElems);
    ei.setStream(PointwiseConvolutionKernel.INPUTS[1], cpuT,
                 cpuT.sizeInBytes() * numWeightsElems);
    ei.setStream(PointwiseConvolutionKernel.INPUTS[2], cpuT,
                 cpuT.sizeInBytes() * numBiasElems);
    ei.setStream(PointwiseConvolutionKernel.OUTPUTS[0], cpuT,
                 cpuT.sizeInBytes() * numOfmapElems);

    return ei;
  }

 public
  static void main(String[] args) {
    PointwiseConvolutionEngineParameters params =
        new PointwiseConvolutionEngineParameters(args);

    PointwiseConvolutionManager mgr = new PointwiseConvolutionManager(params);
    mgr.createSLiCinterface(mgr.interfaceDefault(params));

    BuildConfig cfg = mgr.getBuildConfig();
    cfg.setBuildEffort(Effort.HIGH);
    // cfg.setMPPRCostTableSearchRange(1, 4);
    // cfg.setMPPRParallelism(4);
    mgr.build();
  }
}
