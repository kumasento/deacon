package resnet_18_fst;

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

public class Resnet18FstManager extends Max5LMemManager implements ManagerInterface {
  public Resnet18FstManager(
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
    Resnet18FstEngineParameters params = new Resnet18FstEngineParameters(args);

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
                .Builder(56, 56, 64, 1, 1)
                .BW(8)
                .WBW(2)
                .numFracBits(0)
                .type(Type.IDENTITY)
                .name("identity1")
                .pad(0)
                .stride(1)
                .seq(CompSeq.values()[0])
                .dbg(params.getDebug())
                .coeffOnChip(true)
                .coeffFile(params.getCoeffFile())
                .input("conv0_1")
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
                .numOutputs(1)
                .residual("identity_1")
                .PF(1)
                .PC(1)
                .PK(1)
                .namedRegion("SLR0")
                .build());
            

    Resnet18FstManager mgr = new Resnet18FstManager(params, cps);
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
