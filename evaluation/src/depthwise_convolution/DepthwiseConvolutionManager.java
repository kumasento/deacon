package depthwise_convolution;

import com.custom_computing_ic.maxdeep.kernel.conv2d.DepthwiseConvolutionKernel;
import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.custom_computing_ic.maxdeep.manager.Max5LMemManager;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
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
class DepthwiseConvolutionManager extends Max5LMemManager implements
    ManagerInterface {

 public
  static final String knlName = "depthwise_convolution_kernel";
 private
  final DFEType dfeT;
 private
  final CPUTypes cpuT;

 public
  DepthwiseConvolutionManager(DepthwiseConvolutionEngineParameters params) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

    // dfeT = DFETypeFactory.dfeFloat(8, 24);
    // cpuT = CPUTypes.FLOAT;
    dfeT = DFETypeFactory.dfeFix(6, 10, SignMode.TWOSCOMPLEMENT);
    cpuT = CPUTypes.INT16;

    // constants
    addMaxFileConstant("TILE_HEIGHT", params.getTileHeight());
    addMaxFileConstant("TILE_WIDTH", params.getTileWidth());
    addMaxFileConstant("TILE_DEPTH", params.getTileDepth());
    addMaxFileConstant("KERNEL_SIZE", params.getKernelSize());
    addMaxFileConstant("PAR_WIDTH", params.getParWidth());
    addMaxFileConstant("PAR_DEPTH", params.getParDepth());
    addMaxFileConstant("USE_WINOGRAD", params.getUseWinograd() ? 1 : 0);

    // create a kernel block.
    KernelBlock knl = this.addKernel(new DepthwiseConvolutionKernel(
        this.makeKernelParameters(knlName), params.getTileHeight(),
        params.getTileWidth(), params.getTileDepth(), params.getKernelSize(),
        params.getParWidth(), params.getParDepth(), dfeT, params.getDebug(),
        params.getUseWinograd()));

    // connect them to external environment
    for (String inputName : DepthwiseConvolutionKernel.INPUTS)
      knl.getInput(inputName).connect(this.addStreamFromCPU(inputName));
    for (String outputName : DepthwiseConvolutionKernel.OUTPUTS)
      this.addStreamToCPU(outputName).connect(knl.getOutput(outputName));
  }

 public
  EngineInterface interfaceDefault(
      DepthwiseConvolutionEngineParameters params) {
    EngineInterface ei = new EngineInterface();

    // number of ticks
    InterfaceParam N = ei.addParam("N", CPUTypes.INT64);
    InterfaceParam tileInputHeight =
        ei.addConstant(params.getTileHeight() + params.getKernelSize() - 1);
    InterfaceParam tileInputWidth =
        ei.addConstant(params.getTileWidth() + params.getKernelSize() - 1);
    InterfaceParam numParWidth = tileInputWidth / params.getParWidth();
    InterfaceParam numParDepth =
        ei.addConstant(params.getTileDepth() / params.getParDepth());
    InterfaceParam numTicks = N * numParDepth * numParWidth * tileInputHeight;

    // set ticks
    ei.setTicks(knlName, numTicks);

    // set streams
    InterfaceParam numIfmapElems =
        N * tileInputHeight * tileInputWidth * params.getTileDepth();
    InterfaceParam numOfmapElems =
        N * ei.addConstant(params.getTileHeight() * params.getTileWidth() *
                           params.getTileDepth());
    InterfaceParam numWeightsElems =
        N * ei.addConstant(params.getTileDepth() * params.getKernelSize() *
                           params.getKernelSize());
    InterfaceParam numBiasElems = N * ei.addConstant(params.getTileDepth());

    ei.setStream(DepthwiseConvolutionKernel.INPUTS[0], cpuT,
                 cpuT.sizeInBytes() * numIfmapElems);
    ei.setStream(DepthwiseConvolutionKernel.INPUTS[1], cpuT,
                 cpuT.sizeInBytes() * numWeightsElems);
    ei.setStream(DepthwiseConvolutionKernel.INPUTS[2], cpuT,
                 cpuT.sizeInBytes() * numBiasElems);
    ei.setStream(DepthwiseConvolutionKernel.OUTPUTS[0], cpuT,
                 cpuT.sizeInBytes() * numOfmapElems);

    return ei;
  }

 public
  static void main(String[] args) {
    DepthwiseConvolutionEngineParameters params =
        new DepthwiseConvolutionEngineParameters(args);

    DepthwiseConvolutionManager mgr = new DepthwiseConvolutionManager(params);
    mgr.createSLiCinterface(mgr.interfaceDefault(params));

    BuildConfig cfg = mgr.getBuildConfig();
    cfg.setBuildEffort(Effort.HIGH);
    // cfg.setMPPRCostTableSearchRange(1, 4);
    // cfg.setMPPRParallelism(4);
    mgr.build();
  }
}
