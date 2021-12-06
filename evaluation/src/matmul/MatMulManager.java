package matmul;

import com.custom_computing_ic.maxdeep.kernel.fc.FullyConnectedLayerKernel;
import com.custom_computing_ic.maxdeep.kernel.fc.FullyConnectedLayerParameters;
import com.maxeler.maxcompiler.v2.build.EngineParameters;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

public class MatMulManager extends CustomManager {

  public static final String knlName = "fc";

  public MatMulManager(EngineParameters params, FullyConnectedLayerParameters fp) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);
    config.setDefaultStreamClockFrequency(100);

    // create a kernel block of a fully connected kernel.
    KernelBlock knl = this.addKernel(new FullyConnectedLayerKernel(this
        .makeKernelParameters(knlName), fp));
    // connect them to external environment
    knl.getInput(FullyConnectedLayerKernel.IFMAP_NAME).connect(
        this.addStreamFromCPU("x"));
    knl.getInput(FullyConnectedLayerKernel.COEFF_NAME).connect(
        this.addStreamFromCPU("W"));
    this.addStreamToCPU("y").connect(
        knl.getOutput(FullyConnectedLayerKernel.OFMAP_NAME));
  }

  public EngineInterface interfaceDefault(FullyConnectedLayerParameters fp) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam N = ei.addParam("N", CPUTypes.INT64);

    InterfaceParam ifmapSizeInBytes = N * fp.W * fp.BW / 8;
    InterfaceParam coeffSizeInBytes = N * fp.H * fp.W * fp.BW / 8;
    InterfaceParam ofmapSizeInBytes = N * fp.H * fp.BW / 8;

    ei.setTicks(knlName, N * fp.W * fp.H);
    ei.setStream("x", CPUTypes.UINT8, ifmapSizeInBytes);
    ei.setStream("W", CPUTypes.UINT8, coeffSizeInBytes);
    ei.setStream("y", CPUTypes.UINT8, ofmapSizeInBytes);

    return ei;
  }

  public static void main(String[] args) {
    EngineParameters params = new EngineParameters(args);

    FullyConnectedLayerParameters fp = new FullyConnectedLayerParameters(
        "matmul", 32, 32, 32);

    MatMulManager mgr = new MatMulManager(params, fp);
    mgr.createSLiCinterface(mgr.interfaceDefault(fp));

    BuildConfig cfg = mgr.getBuildConfig();
    cfg.setBuildEffort(Effort.HIGH);
    cfg.setMPPRCostTableSearchRange(1, 4);
    cfg.setMPPRParallelism(4);
    mgr.build();
  }

}
