package mobilenet_v2;

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

public class MobilenetV2Manager extends Max5LMemManager implements ManagerInterface {
  public MobilenetV2Manager(ConvLayerEngineParameters params, List<ConvLayerParameters> cps) {
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
    MobilenetV2EngineParameters params = new MobilenetV2EngineParameters(args);

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();

    cps.add(new ConvLayerParameters.Builder(112, 112, 3, 32, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv0")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(112, 112, 32, 16, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b0c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(112, 112, 16, 16, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b0c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(112, 112, 16, 96, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b1c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 96, 24, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b1c1")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 24, 144, 1)
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
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 144, 24, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b2c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b2c0")

                .residual("b2c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(56, 56, 24, 144, 1)
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
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 144, 32, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b3c1")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 32, 192, 1)
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
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 192, 32, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b4c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b4c0")

                .residual("b4c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 32, 192, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b5c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 192, 32, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b5c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b5c0")

                .residual("b5c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(28, 28, 32, 192, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b6c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 192, 64, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b6c1")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b7c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 64, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b7c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b7c0")

                .residual("b7c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b8c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 64, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b8c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b8c0")

                .residual("b8c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b9c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 64, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b9c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b9c0")

                .residual("b9c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 64, 384, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b10c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 384, 96, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b10c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 96, 576, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b11c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 576, 96, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b11c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b11c0")

                .residual("b11c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 96, 576, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b12c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 576, 96, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b12c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b12c0")

                .residual("b12c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(14, 14, 96, 576, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b13c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 576, 160, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b13c1")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 160, 960, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b14c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 960, 160, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b14c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b14c0")

                .residual("b14c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 160, 960, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b15c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")
                .output(OutputType.OFMAP)
                .output(OutputType.IFMAP)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 960, 160, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b15c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("b15c0")

                .residual("b15c0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 160, 960, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("b16c0")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 960, 320, 3)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.DEPTHWISE_SEPARABLE)
                .name("b16c1")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    cps.add(new ConvLayerParameters.Builder(7, 7, 320, 1280, 1)
                .BW(8)
                .WBW(8)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("convlast")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("")

                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("")
                .pooling(Pooling.MAX)
                .build());

    MobilenetV2Manager mgr = new MobilenetV2Manager(params, cps);
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
