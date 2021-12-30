package resnet_18_onnx_b8_s4;

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

public class Resnet18OnnxB8S4Manager extends Max5LMemManager implements ManagerInterface {
  public Resnet18OnnxB8S4Manager(ConvLayerEngineParameters params, List<ConvLayerParameters> cps) {
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
    Resnet18OnnxB8S4EngineParameters params = new Resnet18OnnxB8S4EngineParameters(args);

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();

    cps.add(new ConvLayerParameters.Builder(112, 112, 3, 64, 7)
                .input("")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15conv0fwd")
                .pad(3)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("resnetv15conv0fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.POOLING)
                .name("resnetv15pool0fwd")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("resnetv15pool0fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage1conv0fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("resnetv15stage1conv0fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage1conv1fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage1conv0fwd_1")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("resnetv15stage1conv1fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage1conv2fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 64, 64, 3)
                .input("resnetv15stage1conv2fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage1conv3fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage1conv2fwd_1")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 64, 128, 3)
                .input("resnetv15stage1conv3fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage2conv0fwd")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 128, 128, 3)
                .input("resnetv15stage2conv0fwd")
                .input("resnetv15stage2conv0fwd_1")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage2conv1fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage2conv0fwd_1")
                .PF(4)
                .PC(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 128, 128, 3)
                .input("resnetv15stage2conv1fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage2conv3fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 128, 128, 3)
                .input("resnetv15stage2conv3fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage2conv4fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage2conv3fwd_1")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 128, 256, 3)
                .input("resnetv15stage2conv4fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage3conv0fwd")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 256, 256, 3)
                .input("resnetv15stage3conv0fwd")
                .input("resnetv15stage3conv0fwd_1")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage3conv1fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage3conv0fwd_1")
                .PF(4)
                .PC(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 256, 256, 3)
                .input("resnetv15stage3conv1fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage3conv3fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 256, 256, 3)
                .input("resnetv15stage3conv3fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage3conv4fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage3conv3fwd_1")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 256, 512, 3)
                .input("resnetv15stage3conv4fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage4conv0fwd")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 512, 512, 3)
                .input("resnetv15stage4conv0fwd")
                .input("resnetv15stage4conv0fwd_1")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage4conv1fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage4conv0fwd_1")
                .PF(4)
                .PC(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 512, 512, 3)
                .input("resnetv15stage4conv1fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage4conv3fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(4)
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 512, 512, 3)
                .input("resnetv15stage4conv3fwd")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("resnetv15stage4conv4fwd")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("resnetv15stage4conv3fwd_1")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    Resnet18OnnxB8S4Manager mgr = new Resnet18OnnxB8S4Manager(params, cps);
    mgr.createSLiCinterface(mgr.interfaceDefault(cps, params));
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.VERY_HIGH);
    buildConfig.setOptimizationGoal(BuildConfig.OptimizationGoal.BALANCED);
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
