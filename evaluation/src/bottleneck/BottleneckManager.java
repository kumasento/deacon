package bottleneck;

import java.util.ArrayList;
import java.util.List;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.custom_computing_ic.maxdeep.manager.ManagerUtils;
import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.custom_computing_ic.maxdeep.manager.CustomLMemManager;
import com.custom_computing_ic.maxdeep.manager.Max5LMemManager;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.BuildConfig.Effort;
import com.maxeler.platform.max5.manager.BuildConfig.OptimizationGoal;

/**
 * The manager that builds three consecutive Convolution layers following the
 * bottleneck style in ResNet.
 *
 * This has been adapted to Max5.
 *
 * @author Ruizhe Zhao
 *
 */
public class BottleneckManager extends Max5LMemManager implements ManagerInterface {
  public BottleneckManager(BottleneckEngineParameters params,
      List<ConvLayerParameters> cps) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

    ConvLayerManagerUtils.createKernelBlocks(this, cps, /* numCoeffFifoSplits= */1, params.getUseDRAM());
    ConvLayerManagerUtils.setupConstants(this, cps, params);
  }

  public EngineInterface interfaceDefault(List<ConvLayerParameters> cps,
      BottleneckEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    ConvLayerManagerUtils.setupStreams(ei, cps, batchSize, ep.getUseDRAM());
    ManagerUtils.ignoreLMemStreams(ei);

    return ei;
  }

  @SuppressWarnings("deprecation")
  public static void main(String[] args) {
    BottleneckEngineParameters ep = new BottleneckEngineParameters(args);
    int H = ep.getH();
    int W = ep.getW();
    int C = ep.getC();
    int F = ep.getF();
    int K = ep.getK();

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();

    /* resolve the computation sequence */
    CompSeq seq0 = CompSeq.values()[ep.getSeq0()];
    CompSeq seq1 = CompSeq.values()[ep.getSeq1()];
    CompSeq seq2 = CompSeq.values()[ep.getSeq2()];

    // Pointwise
    cps.add(0, new ConvLayerParameters.Builder(H, W, C, F, 1)
        .name("conv0")
        .BW(ep.getBitWidth())
        .PC(ep.getPC())
        .PF(ep.getPF())
        .PK(1)
        .pad(0)
        .seq(seq0)
        .type(Type.POINTWISE)
        .coeffOnChip(ep.getCoeffOnChip())
        // .dbg(true)
        .build());
    // Standard
    cps.add(1, new ConvLayerParameters.Builder(H - K + 1, W - K + 1, F, F, K)
        .name("conv1")
        .BW(ep.getBitWidth())
        .PC(ep.getPF())
        .PF(ep.getPF())
        .PK(ep.getPK())
        .pad(0)
        .seq(seq1)
        // .dbg(true)
        .coeffOnChip(ep.getCoeffOnChip())
        .build());
    // Pointwise
    cps.add(2, new ConvLayerParameters.Builder(H - K + 1, W - K + 1, F, F, 1)
        .name("conv2")
        .BW(ep.getBitWidth())
        .PC(ep.getPC())
        .PF(ep.getPF())
        .PK(1)
        .pad(0)
        .seq(seq2)
        // .dbg(true)
        .type(Type.POINTWISE)
        .coeffOnChip(ep.getCoeffOnChip())
        .build());

    BottleneckManager mgr = new BottleneckManager(ep, cps);

    mgr.createSLiCinterface(mgr.interfaceDefault(cps, ep));
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    mgr.build();
  }
}
