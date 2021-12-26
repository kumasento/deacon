package mobilenet_v1_par_s16;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.OutputType;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Pooling;
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

public class MobilenetV1ParS16Manager extends Max5LMemManager implements ManagerInterface {
  public MobilenetV1ParS16Manager(ConvLayerEngineParameters params, List<ConvLayerParameters> cps) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

    ConvLayerManagerUtils.createKernelBlocks(this, cps, /* numCoeffFifoSplits= */ 1, true);
    ConvLayerManagerUtils.setupConstants(this, cps, params);
  }

  public EngineInterface interfaceDefault(
      List<ConvLayerParameters> cps, ConvLayerEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    ConvLayerManagerUtils.setupStreams(ei, cps, batchSize, true, this);
    ManagerUtils.ignoreLMemStreams(ei);

    return ei;
  }

  @SuppressWarnings("deprecation")
  public static void main(String[] args) {
    MobilenetV1ParS16EngineParameters params = new MobilenetV1ParS16EngineParameters(args);

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();

    cps.add(new ConvLayerParameters.Builder(112, 112, 3, 32, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv0")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(112, 112, 32, 64, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 128, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv2")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 128, 128, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv3")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 128, 256, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv4")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 256, 256, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv5")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 256, 512, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv6")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 512, 512, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv7")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 512, 512, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv8")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 512, 512, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv9")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 512, 512, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv10")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 512, 512, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv11")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 512, 1024, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv12")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 1024, 1024, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv13")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(16)
                .PC(16)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    MobilenetV1ParS16Manager mgr = new MobilenetV1ParS16Manager(params, cps);
    mgr.createSLiCinterface(mgr.interfaceDefault(cps, params));
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.VERY_HIGH);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER1);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER2);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER3);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER4);
    buildConfig.addImplementationStrategy(ImplementationStrategy.PERFORMANCE_EARLY_BLOCK_PLACEMENT);
    buildConfig.addImplementationStrategy(ImplementationStrategy.PERFORMANCE_EXPLORE);
    buildConfig.addImplementationStrategy(ImplementationStrategy.PERFORMANCE_EXTRA_TIMING_OPT);
    buildConfig.addImplementationStrategy(ImplementationStrategy.PERFORMANCE_NET_DELAY_HIGH);
    buildConfig.addImplementationStrategy(ImplementationStrategy.PERFORMANCE_REFINE_PLACEMENT);
    buildConfig.setParallelism(4);

    mgr.build();
  }
}
