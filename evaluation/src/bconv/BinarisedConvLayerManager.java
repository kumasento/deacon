package bconv;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerWrapKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.BinaryPackKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.BinaryUnpackKernel;
import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

/**
 * 
 */

/**
 * This manager for binarised convolution layer doesn't rely on the
 * ConvLayerManagerUtils to create kernels at the moment.
 * 
 * @author Ruizhe Zhao
 * 
 */
public class BinarisedConvLayerManager extends CustomManager {

  public BinarisedConvLayerManager(ConvLayerEngineParameters params,
      ConvLayerParameters cp) {
    super(params);

    if (params.getBitWidth() != 1)
      throw new IllegalArgumentException("BW should equal 1");

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);
    config.setDefaultStreamClockFrequency(params.getFreq());

    createKernelBlocks(cp);

    // ConvLayerManagerUtils.createKernelBlocks(this, cp, params.getUseDRAM());
    ConvLayerManagerUtils.setupConstants(this, cp, params);
  }

  public void createKernelBlocks(ConvLayerParameters cp) {
    KernelBlock knl =
        this.addKernel(new ConvLayerWrapKernel(this
            .makeKernelParameters(cp.name), cp));

    KernelBlock unpackIfmap =
        this.addKernel(new BinaryUnpackKernel(this
            .makeKernelParameters("unpack_ifmap")));
    KernelBlock unpackCoeff =
        this.addKernel(new BinaryUnpackKernel(this
            .makeKernelParameters("unpack_coeff")));
    KernelBlock packOfmap =
        this.addKernel(new BinaryPackKernel(this
            .makeKernelParameters("pack_ofmap")));

    knl.getInput(ConvLayerWrapKernel.IFMAP_NAME).connect(
        unpackIfmap.getOutput(BinaryUnpackKernel.OUT_NAME));
    knl.getInput(ConvLayerWrapKernel.COEFF_NAME).connect(
        unpackCoeff.getOutput(BinaryUnpackKernel.OUT_NAME));
    packOfmap.getInput(BinaryPackKernel.INP_NAME).connect(
        knl.getOutput(ConvLayerWrapKernel.OFMAP_NAME));

    unpackIfmap.getInput(BinaryUnpackKernel.INP_NAME).connect(
        this.addStreamFromCPU("ifmap"));
    unpackCoeff.getInput(BinaryUnpackKernel.INP_NAME).connect(
        this.addStreamFromCPU("coeff"));
    this.addStreamToCPU("ofmap").connect(
        packOfmap.getOutput(BinaryPackKernel.OUT_NAME));
  }

  public void setupStreams(EngineInterface ei, ConvLayerParameters cp,
      InterfaceParam batchSize) {

    if (cp.getIfmapStreamNumElems() % 8 != 0)
      throw new IllegalArgumentException(
          "ifmap stream elements should be 8's multiple");
    if (cp.getCoeffStreamNumElems() % 8 != 0)
      throw new IllegalArgumentException(
          "coeff stream elements should be 8's multiple");
    if (cp.getOfmapStreamNumElems() % 8 != 0)
      throw new IllegalArgumentException(
          "ofmap stream elements should be 8's multiple");

    ei.setTicks(cp.name, cp.getNumCycles() * batchSize);

    ei.setTicks("unpack_ifmap", cp.getIfmapStreamNumElems() * batchSize / 8);
    ei.setTicks("unpack_coeff", cp.getCoeffStreamNumElems() * batchSize / 8);
    ei.setTicks("pack_ofmap", cp.getOfmapStreamNumElems() * batchSize / 8);

    ei.setStream("ifmap", cp.getCPUTypes(), cp.getIfmapStreamNumElems()
        * batchSize / 8);
    ei.setStream("coeff", cp.getCPUTypes(), cp.getCoeffStreamNumElems()
        * batchSize / 8);
    ei.setStream("ofmap", cp.getCPUTypes(), cp.getOfmapStreamNumElems()
        * batchSize / 8);
  }

  public EngineInterface interfaceDefault(ConvLayerParameters cp,
      ConvLayerEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    // ConvLayerManagerUtils.setupStreams(ei, cp, batchSize, ep.getUseDRAM());
    // ConvLayerManagerUtils.setupKernelTicks(ei, cp, batchSize);

    setupStreams(ei, cp, batchSize);

    return ei;
  }

  @SuppressWarnings("deprecation")
  public static void main(String[] args) {
    BinarisedConvLayerEngineParameters params =
        new BinarisedConvLayerEngineParameters(args);

    int H = params.getH();
    int W = params.getW();
    int C = params.getC();
    int F = params.getF();
    int K = params.getK();

    ConvLayerParameters cp =
        new ConvLayerParameters.Builder(H, W, C, F, K)
            .name("conv").PC(params.getPC()).PF(params.getPF())
            .PK(params.getPK()).BW(params.getBitWidth())
            .seq(CompSeq.values()[params.getSeq()]).build();

    BinarisedConvLayerManager mgr = new BinarisedConvLayerManager(params, cp);
    mgr.createSLiCinterface(mgr.interfaceDefault(cp, params));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    buildConfig.setMPPRCostTableSearchRange(1, 4);
    buildConfig.setMPPRParallelism(4);
    mgr.build();
  }
}
