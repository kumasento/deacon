package mobilenet_v2_onnx;

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

public class MobilenetV2OnnxManager extends Max5LMemManager implements ManagerInterface {
  public MobilenetV2OnnxManager(ConvLayerEngineParameters params, List<ConvLayerParameters> cps) {
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
    MobilenetV2OnnxEngineParameters params = new MobilenetV2OnnxEngineParameters(args);

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();

    cps.add(new ConvLayerParameters.Builder(112, 112, 3, 32, 3)
                .input("")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv0")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(112, 112, 32, 16, 3)
                .input("conv0")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv2")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(112, 112, 16, 96, 1)
                .input("conv2")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv5")
                .pad(0)
                .stride(1)
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 96, 24, 3)
                .input("conv5")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv7")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 24, 144, 1)
                .input("conv7")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv10")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 144, 24, 3)
                .input("conv10")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv12")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv10_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 24, 144, 1)
                .input("conv12")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv16")
                .pad(0)
                .stride(1)
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 144, 32, 3)
                .input("conv16")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv18")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 32, 192, 1)
                .input("conv18")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv21")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 192, 32, 3)
                .input("conv21")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv23")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv21_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 32, 192, 1)
                .input("conv23")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv27")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 192, 32, 3)
                .input("conv27")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv29")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv27_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 32, 192, 1)
                .input("conv29")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv33")
                .pad(0)
                .stride(1)
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 192, 64, 3)
                .input("conv33")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv35")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .input("conv35")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv38")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 64, 3)
                .input("conv38")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv40")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv38_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .input("conv40")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv44")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 64, 3)
                .input("conv44")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv46")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv44_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .input("conv46")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv50")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 64, 3)
                .input("conv50")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv52")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv50_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .input("conv52")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv56")
                .pad(0)
                .stride(1)
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 96, 3)
                .input("conv56")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv58")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 96, 576, 1)
                .input("conv58")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv61")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 576, 96, 3)
                .input("conv61")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv63")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv61_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 96, 576, 1)
                .input("conv63")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv67")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 576, 96, 3)
                .input("conv67")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv69")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv67_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 96, 576, 1)
                .input("conv69")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv73")
                .pad(0)
                .stride(1)
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 576, 160, 3)
                .input("conv73")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv75")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 160, 960, 1)
                .input("conv75")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv78")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 960, 160, 3)
                .input("conv78")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv80")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv78_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 160, 960, 1)
                .input("conv80")
                .output(new Output(OutputType.OFMAP, 0))
                .output(new Output(OutputType.IFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv84")
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 960, 160, 3)
                .input("conv84")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv86")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("conv84_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 160, 960, 1)
                .input("conv86")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv90")
                .pad(0)
                .stride(1)
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
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 960, 320, 3)
                .input("conv90")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("conv92")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .dspFactor(0.5)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 320, 1280, 1)
                .input("conv92")
                .output(new Output(OutputType.OFMAP, 0))
                .BW(16)
                .WBW(16)
                .numFracBits(8)
                .type(Type.STANDARD)
                .name("conv95")
                .pad(0)
                .stride(1)
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
                .dspFactor(0.5)
                .build());

    MobilenetV2OnnxManager mgr = new MobilenetV2OnnxManager(params, cps);
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
