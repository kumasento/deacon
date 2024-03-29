package conv_single_layer;

import com.custom_computing_ic.dfe_snippets.manager.ManagerUtils;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.custom_computing_ic.maxdeep.manager.CustomLMemManager;
import com.custom_computing_ic.maxdeep.manager.conv_single_layer.ConvSingleLayerEngineParameters;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.OptimizationGoal;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

public class ConvSingleLayerManager extends CustomLMemManager {
  public ConvSingleLayerManager(ConvSingleLayerEngineParameters params,
      ConvLayerParameters cp) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);
    config.setDefaultStreamClockFrequency(params.getFreq());

    ConvLayerManagerUtils.createKernelBlocks(
        this, cp, params.getNumCoeffFifoSplits(), params.getUseDRAM());

    ConvLayerManagerUtils.setupConstants(this, cp, params);
  }

  public EngineInterface interfaceDefault(ConvLayerParameters cp,
      ConvLayerEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    // number of tiles with indentical sizes to be processed
    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    // setup streams of the convolution layer defined cp
    ConvLayerManagerUtils.setupStreams(ei, cp, batchSize, ep.getUseDRAM(),
        this);
    ManagerUtils.ignoreLMemStreams(ei);

    return ei;
  }

  public static void main(String[] args) {
    ConvSingleLayerEngineParameters params = new ConvSingleLayerEngineParameters(args);

    // Get the configuration for building a tile in the hardware
    int H = params.getH();
    int W = params.getW();
    int C = params.getC();
    int F = params.getF();
    int K = params.getK();

    ConvLayerParameters cp = new ConvLayerParameters.Builder(H, W, C, F, K)
        .name("conv")
        .PC(params.getPC())
        .PF(params.getPF())
        .PK(params.getPK())
        .BW(params.getBitWidth())
        .dtype(params.getDType())
        .numFracBits(params.getNumFracBits())
        .seq(CompSeq.values()[params.getSeq()])
        .dbg(params.getDebug())
        .useWinograd(params.getUseWinograd())
        .winogradWeightsOffline(params.getWinogradWeightsOffline())
        .coeffOnChip(params.getCoeffOnChip())
        .build();

    ConvSingleLayerManager mgr = new ConvSingleLayerManager(params, cp);
    mgr.createSLiCinterface(mgr.interfaceDefault(cp, params));
    // iface should be accessible from the parent class
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.VERY_HIGH);
    buildConfig.setMPPRCostTableSearchRange(1, 16);
    buildConfig.setOptimizationGoal(OptimizationGoal.SPEED);
    buildConfig.setMPPRParallelism(4);

    mgr.build();
  }
}
