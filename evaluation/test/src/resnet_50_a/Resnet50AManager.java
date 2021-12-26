package resnet_50_a;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Output;
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

public class Resnet50AManager extends Max5LMemManager implements ManagerInterface {
  public Resnet50AManager(ConvLayerEngineParameters params, List<ConvLayerParameters> cps) {
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
    Resnet50AEngineParameters params = new Resnet50AEngineParameters(args);

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();

    cps.add(new ConvLayerParameters.Builder(112, 112, 3, 64, 7)
                .input("")
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv0")
                .pad(2)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("")
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.POOLING)
                .name("pool0")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 1)
                .input("")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b2c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("b2c0")
                .input("b2c0_1")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 1))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b2c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PF(1)
                .PC(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 256, 1)
                .input("b2c1")
                .input("b2c1_1")
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b2c2")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("b2c1_1")
                .PF(1)
                .PC(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 256, 64, 1)
                .input("")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b3c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("b3c0")
                .input("b3c0_1")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 1))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b3c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PF(4)
                .PC(1)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 256, 1)
                .input("b3c1")
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b3c2")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("b3c1_1")
                .PF(4)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 256, 64, 1)
                .input("")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b4c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("b4c0")
                .input("b4c0_1")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 1))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b4c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PF(4)
                .PC(1)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 256, 1)
                .input("b4c1")
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b4c2")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("b4c1_1")
                .PF(4)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    Resnet50AManager mgr = new Resnet50AManager(params, cps);
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
