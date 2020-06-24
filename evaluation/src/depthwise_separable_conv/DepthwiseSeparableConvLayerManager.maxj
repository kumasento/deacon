package depthwise_separable_conv;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.custom_computing_ic.maxdeep.manager.Max5LMemManager;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.BuildConfig.Effort;
import com.maxeler.platform.max5.manager.BuildConfig.OptimizationGoal;

public
class DepthwiseSeparableConvLayerManager extends Max5LMemManager implements
    ManagerInterface {

 public
  DepthwiseSeparableConvLayerManager(ConvLayerEngineParameters params,
                                     ConvLayerParameters cp) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

    ConvLayerManagerUtils.createKernelBlocks(this, cp, params.getUseDRAM());
    ConvLayerManagerUtils.setupConstants(this, cp, params);
  }

 public
  EngineInterface interfaceDefault(ConvLayerParameters cp,
                                   ConvLayerEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    ConvLayerManagerUtils.setupStreams(ei, cp, batchSize, ep.getUseDRAM(),
                                       this);
    // ConvLayerManagerUtils.setupKernelTicks(ei, cp, batchSize);

    return ei;
  }

  @SuppressWarnings("deprecation") public static void main(String[] args) {
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
            "Version number is not recognizable " +
            Integer.toString(params.getVersion()));
    }

    ConvLayerParameters cp =
        new ConvLayerParameters
            .Builder(params.getH(), params.getW(), params.getC(), params.getF(),
                     params.getK())
            .type(type)
            .seq(CompSeq.FILTER_MAJOR)
            .name("conv")
            .PC(params.getPC())
            .PF(params.getPF())
            .PK(params.getPK())
            .BW(params.getBitWidth())
            .dbg(false)
            .build();

    DepthwiseSeparableConvLayerManager mgr =
        new DepthwiseSeparableConvLayerManager(params, cp);
    mgr.createSLiCinterface(mgr.interfaceDefault(cp, params));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    // buildConfig.setMPPRCostTableSearchRange(1, 4);
    // buildConfig.setMPPRParallelism(4);
    mgr.build();
  }
}
