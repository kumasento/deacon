package conv_pad;

import java.util.ArrayList;
import java.util.List;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
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
 * The manager that builds two consecutive Convolution layers.
 *
 * This has been adapted to Max5.
 *
 * @author Ruizhe Zhao
 *
 */
public class ConvPadManager extends Max5LMemManager implements ManagerInterface {
  public ConvPadManager(ConvPadEngineParameters params,
      List<ConvLayerParameters> cps) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    this.setAllowNonMultipleTransitions(true);
    this.setDefaultStreamClockFrequency(params.getFreq());

    ConvLayerManagerUtils.createKernelBlocks(this, cps, 1, params.getUseDRAM());
    ConvLayerManagerUtils.setupConstants(this, cps, params);
  }

  public EngineInterface interfaceDefault(List<ConvLayerParameters> cps,
      ConvPadEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    ConvLayerManagerUtils.setupStreams(ei, cps, batchSize, ep.getUseDRAM());
    ManagerUtils.ignoreLMemStreams(ei);

    return ei;
  }

  @SuppressWarnings("deprecation")
  public static void main(String[] args) {
    ConvPadEngineParameters ep = new ConvPadEngineParameters(args);
    int H = ep.getH();
    int W = ep.getW();
    int C = ep.getC();
    int F = ep.getF();
    int K = ep.getK();
    int pad = ep.getPad();

    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();

    /* resolve the computation sequence */
    CompSeq seq = CompSeq.values()[ep.getSeq()];

    cps.add(0, new ConvLayerParameters.Builder(H, W, C, F, K)
        .BW(ep.getBitWidth())
        .name("conv0")
        .PC(ep.getPC())
        .PF(ep.getPF())
        .PK(ep.getPK())
        .pad(ep.getPad())
        .seq(seq)
        .dbg(ep.getDebug())
        .coeffOnChip(ep.getCoeffOnChip())
        .build());

    ConvPadManager mgr = new ConvPadManager(ep, cps);

    mgr.createSLiCinterface(mgr.interfaceDefault(cps, ep));
    mgr.createSLiCinterface(ManagerUtils.dramRead(mgr, mgr.iface));
    mgr.createSLiCinterface(ManagerUtils.dramWrite(mgr, mgr.iface));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    mgr.build();
  }
}
