package conv_single_layer;

import com.custom_computing_ic.maxdeep.manager.ManagerUtils;
import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.custom_computing_ic.maxdeep.manager.Max5LMemManager;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.custom_computing_ic.maxdeep.manager.conv_single_layer.ConvSingleLayerEngineParameters;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemInterface;
import com.maxeler.platform.max5.manager.Max5LimaManager;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.BuildConfig.Effort;
import com.maxeler.platform.max5.manager.BuildConfig.OptimizationGoal;
import com.custom_computing_ic.maxdeep.manager.CustomLMemManager;

public class ConvSingleLayerMax5LimaManager extends Max5LMemManager implements
    ManagerInterface {
  public ConvSingleLayerMax5LimaManager(ConvSingleLayerEngineParameters params,
      ConvLayerParameters cp) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

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

  @SuppressWarnings("deprecation")
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

    ConvSingleLayerMax5LimaManager mgr = new ConvSingleLayerMax5LimaManager(params, cp);
    mgr.createSLiCinterface(mgr.interfaceDefault(cp, params));
    // iface should be accessible from the parent class
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.VERY_HIGH);
    buildConfig.setOptimizationGoal(OptimizationGoal.SPEED);

    mgr.build();
  }
}
