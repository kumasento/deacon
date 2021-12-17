package depthwise_separable_conv;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.custom_computing_ic.maxdeep.manager.ManagerUtils;
import com.custom_computing_ic.maxdeep.manager.Max5LMemManager;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.BuildConfig.Effort;
import com.maxeler.platform.max5.manager.BuildConfig.OptimizationGoal;
import com.maxeler.platform.max5.manager.ImplementationStrategy;
import java.util.ArrayList;
import java.util.List;

public class DepthwiseSeparableConvLayerManager
    extends Max5LMemManager implements ManagerInterface {
  public DepthwiseSeparableConvLayerManager(
      ConvLayerEngineParameters params, List<ConvLayerParameters> cps) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

    ConvLayerManagerUtils.createKernelBlocks(
        this, cps, /* numCoeffFifoSplits= */ 1, params.getUseDRAM());
    ConvLayerManagerUtils.setupConstants(this, cps, params);
  }

  public EngineInterface interfaceDefault(
      List<ConvLayerParameters> cps, ConvLayerEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    ConvLayerManagerUtils.setupStreams(ei, cps, batchSize, ep.getUseDRAM(), this);
    ManagerUtils.ignoreLMemStreams(ei);
    // ConvLayerManagerUtils.setupKernelTicks(ei, cp, batchSize);

    return ei;
  }

  @SuppressWarnings("deprecation")
  public static void main(String[] args) {
    DepthwiseSeparableConvLayerEngineParameters params =
        new DepthwiseSeparableConvLayerEngineParameters(args);

    Type type;
    switch (params.getVersion()) {
      case 1:
        type = Type.DEPTHWISE_SEPARABLE;
        break;
      case 2:
        type = Type.DEPTHWISE_SEPARABLE_V2;
        break;
      default:
        throw new IllegalArgumentException(
            "Version number is not recognizable " + Integer.toString(params.getVersion()));
    }

    CompSeq seq = CompSeq.values()[params.getSeq()];

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();
    for (int i = 0; i < params.getNumLayer(); ++i) {
      ConvLayerParameters cp =
          new ConvLayerParameters
              .Builder(params.getH(), params.getW(), params.getC(), params.getF(), params.getK())
              .BW(params.getBitWidth())
              .WBW(params.getWBW())
              .numFracBits(params.getNumFracBits())
              .type(type)
              .seq(seq)
              .name("conv" + Integer.toString(i))
              .pad(1)
              .stride(i == 0 ? params.getStride() : 1)
              .PC(i == 0 ? params.getPC() : params.getPF())
              .PF(params.getPF())
              .PK(params.getPK())
              .dbg(params.getDebug())
              .coeffOnChip(params.getCoeffOnChip())
              .coeffFile(params.getCoeffFile())
              .build();
      cps.add(cp);
    }

    DepthwiseSeparableConvLayerManager mgr = new DepthwiseSeparableConvLayerManager(params, cps);
    mgr.createSLiCinterface(mgr.interfaceDefault(cps, params));
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER1);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER2);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER3);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER4);
    buildConfig.setParallelism(5);

    mgr.build();
  }
}
