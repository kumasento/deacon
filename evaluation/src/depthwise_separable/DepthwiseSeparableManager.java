package depthwise_separable;

import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFETypeFactory;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

public class DepthwiseSeparableManager extends CustomManager {

  public static final String knlName = "depthwise_separable_kernel";
  private final DFEType dfeT;
  private final CPUTypes cpuT;

  public DepthwiseSeparableManager(DepthwiseSeparableEngineParameters p) {
    super(p);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);
    config.setDefaultStreamClockFrequency(p.getFreq());

    dfeT = DFETypeFactory.dfeFloat(8, 24);
    cpuT = CPUTypes.FLOAT;


    // constants
    addMaxFileConstant("TILE_HEIGHT", p.getTileHeight());
    addMaxFileConstant("TILE_WIDTH", p.getTileWidth());
    addMaxFileConstant("TILE_IN_DEPTH", p.getTileInDepth());
    addMaxFileConstant("TILE_OUT_DEPTH", p.getTileOutDepth());
    addMaxFileConstant("KERNEL_SIZE", p.getKernelSize());
    addMaxFileConstant("PAR_WIDTH", p.getParWidth());
    addMaxFileConstant("PAR_IN_DEPTH", p.getParInDepth());
    addMaxFileConstant("PAR_OUT_DEPTH", p.getParOutDepth());


    // create a kernel block.
    KernelBlock knl =
        this.addKernel(new DepthwiseSeparableKernel(this.makeKernelParameters(knlName), p
            .getTileHeight(), p.getTileWidth(), p.getTileInDepth(), p.getTileOutDepth(), p
            .getKernelSize(), 1, p.getParWidth(), p.getParInDepth(), p.getParOutDepth(), dfeT, p
            .getDebug()));

    // connect them to external environment
    for (String inputName : DepthwiseSeparableKernel.INPUTS)
      knl.getInput(inputName).connect(this.addStreamFromCPU(inputName));
    for (String outputName : DepthwiseSeparableKernel.OUTPUTS)
      this.addStreamToCPU(outputName).connect(knl.getOutput(outputName));
  }

  public EngineInterface interfaceDefault(DepthwiseSeparableEngineParameters params) {
    EngineInterface ei = new EngineInterface();

    // number of ticks
    InterfaceParam N = ei.addParam("N", CPUTypes.INT64);
    InterfaceParam tileInputHeight =
        ei.addConstant(params.getTileHeight() + params.getKernelSize() - 1);
    InterfaceParam tileInputWidth =
        ei.addConstant(params.getTileWidth() + params.getKernelSize() - 1);
    InterfaceParam numParWidth = tileInputWidth / params.getParWidth();
    InterfaceParam numParInDepth = ei.addConstant(params.getTileInDepth() / params.getParInDepth());
    InterfaceParam numParOutDepth =
        ei.addConstant(params.getTileOutDepth() / params.getParOutDepth());

    InterfaceParam numTicks = N * numParInDepth * numParOutDepth * numParWidth * tileInputHeight;

    // set ticks
    ei.setTicks(knlName, numTicks);

    // set streams
    InterfaceParam numIfmapElems = N * tileInputHeight * tileInputWidth * params.getTileInDepth();
    InterfaceParam numOfmapElems =
        ei.addConstant(params.getTileHeight() * params.getTileWidth() * params.getTileOutDepth())
            * N;
    InterfaceParam numDepthwiseWeightsElems =
        ei.addConstant(params.getTileInDepth() * params.getKernelSize() * params.getKernelSize())
            * N;
    InterfaceParam numPointwiseWeightsElems =
        ei.addConstant(params.getTileOutDepth() * params.getTileInDepth()) * N;

    ei.setStream(DepthwiseSeparableKernel.INPUTS[0], cpuT, cpuT.sizeInBytes() * numIfmapElems);
    ei.setStream(DepthwiseSeparableKernel.INPUTS[1], cpuT, cpuT.sizeInBytes()
        * numDepthwiseWeightsElems);
    ei.setStream(DepthwiseSeparableKernel.INPUTS[2], cpuT, cpuT.sizeInBytes()
        * numPointwiseWeightsElems);
    ei.setStream(DepthwiseSeparableKernel.OUTPUTS[0], cpuT, cpuT.sizeInBytes() * numOfmapElems);

    return ei;
  }

  public static void main(String[] args) {
    DepthwiseSeparableEngineParameters params = new DepthwiseSeparableEngineParameters(args);

    DepthwiseSeparableManager mgr = new DepthwiseSeparableManager(params);
    mgr.createSLiCinterface(mgr.interfaceDefault(params));

    BuildConfig cfg = mgr.getBuildConfig();
    cfg.setBuildEffort(Effort.HIGH);
    cfg.setMPPRCostTableSearchRange(1, 4);
    cfg.setMPPRParallelism(4);
    mgr.build();
  }

}
