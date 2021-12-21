package resnet_18;

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

public class Resnet18Manager extends Max5LMemManager implements ManagerInterface {
  public Resnet18Manager(
      ConvLayerEngineParameters params, List<ConvLayerParameters> cps) {
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
    Resnet18EngineParameters params = new Resnet18EngineParameters(args);

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();
    
    cps.add(new ConvLayerParameters
                .Builder(112, 112, 3, 64, 3)
                .BW(8)
                .WBW(2)
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
                .numOutputs(2)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR0")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(56, 56, 64, 64, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv1")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[1])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv0")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR0")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(56, 56, 64, 64, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv2")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv1")
                .numOutputs(2)
                .residual("conv0_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR0")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(56, 56, 64, 64, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv3")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv2")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR0")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(56, 56, 64, 64, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv4")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv3")
                .numOutputs(2)
                .residual("conv2_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR0")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(28, 28, 64, 128, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv5")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv4")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR0")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(28, 28, 64, 128, 1)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.POINTWISE)
                .name("shortcut2")
                .pad(0)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv4_1")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(28, 28, 128, 128, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv6")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv5")
                .numOutputs(2)
                .residual("shortcut2")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(28, 28, 128, 128, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv7")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv6")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(28, 28, 128, 128, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv8")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv7")
                .numOutputs(2)
                .residual("conv6_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(14, 14, 128, 256, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv9")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv8")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(14, 14, 128, 256, 1)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.POINTWISE)
                .name("shortcut3")
                .pad(0)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv8_1")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(14, 14, 256, 256, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv10")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv9")
                .numOutputs(2)
                .residual("shortcut3")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(14, 14, 256, 256, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv11")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv10")
                .numOutputs(1)
                .residual("")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(14, 14, 256, 256, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv12")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv11")
                .numOutputs(2)
                .residual("conv10_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR1")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(7, 7, 256, 512, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv13")
                .pad(1)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv12")
                .numOutputs(1)
                .residual("")
                .PF(4)
                .PC(1)
                .PK(1)
                .namedRegion("SLR2")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(7, 7, 256, 512, 1)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.POINTWISE)
                .name("shortcut4")
                .pad(0)
                .stride(2)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv12_1")
                .numOutputs(1)
                .residual("")
                .PF(4)
                .PC(1)
                .PK(1)
                .namedRegion("SLR2")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(7, 7, 512, 512, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv14")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv13")
                .numOutputs(2)
                .residual("shortcut4")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("SLR2")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(7, 7, 512, 512, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv15")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv14")
                .numOutputs(1)
                .residual("")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("SLR2")
                .build());
            
    cps.add(new ConvLayerParameters
                .Builder(7, 7, 512, 512, 3)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.STANDARD)
                .name("conv16")
                .pad(1)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv15")
                .numOutputs(1)
                .residual("conv14_1")
                .PF(4)
                .PC(4)
                .PK(1)
                .namedRegion("SLR2")
                .build());
            

    Resnet18Manager mgr = new Resnet18Manager(params, cps);
    mgr.createSLiCinterface(mgr.interfaceDefault(cps, params));
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.VERY_HIGH);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER1);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER2);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER3);
    buildConfig.addImplementationStrategy(ImplementationStrategy.MAXELER4);
    buildConfig.setParallelism(5);

    mgr.build();
  }
}
