package conv;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.manager.ConvLayerEngineParameters;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

/**
 * 
 */

/**
 * @author Ruizhe Zhao
 * 
 */
public class ConvLayerManager extends CustomManager {

  public ConvLayerManager(ConvLayerEngineParameters params, ConvLayerParameters cp) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);

    ConvLayerManagerUtils.createKernelBlocks(this, cp, params.getUseDRAM());
    ConvLayerManagerUtils.setupConstants(this, cp, params);
  }

  public EngineInterface interfaceDefault(ConvLayerParameters cp, ConvLayerEngineParameters ep) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    ConvLayerManagerUtils.setupStreams(ei, cp, batchSize, ep.getUseDRAM());
    // ConvLayerManagerUtils.setupKernelTicks(ei, cp, batchSize);

    return ei;
  }

  @SuppressWarnings("deprecation")
  public static void main(String[] args) {
    ConvLayerEngineParameters params = new ConvLayerEngineParameters(args);

    int H = params.getH();
    int W = params.getW();
    int C = params.getC();
    int F = params.getF();
    int K = params.getK();

    ConvLayerParameters cp =
        new ConvLayerParameters.Builder(H, W, C, F, K).name("conv").PC(params.getPC())
            .PF(params.getPF()).PK(params.getPK()).BW(params.getBitWidth())
            .seq(CompSeq.values()[params.getSeq()]).build();

    ConvLayerManager mgr = new ConvLayerManager(params, cp);
    mgr.createSLiCinterface(mgr.interfaceDefault(cp, params));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    buildConfig.setMPPRCostTableSearchRange(1, 4);
    buildConfig.setMPPRParallelism(4);
    mgr.build();
  }
}
