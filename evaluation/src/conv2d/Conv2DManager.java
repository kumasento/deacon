package conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;

/**
 * 
 */

/**
 * @author Ruizhe Zhao
 * 
 */
public class Conv2DManager extends CustomManager {

  public static final int H           = 32;
  public static final int W           = 32;
  public static final int C           = 32;
  public static final int F           = 32;
  public static final int K           = 3;

  private final String    KERNEL_NAME = "Conv2DWrapKernel";

  public Conv2DManager(Conv2DEngineParameters params, ConvLayerParameters convParams) {
    super(params);

    getCurrentKernelConfig().debug.setEnableLatencyAnnotation(true);
    config.setAllowNonMultipleTransitions(true);

    KernelBlock knl = addKernel(new Conv2DWrapKernel(
        makeKernelParameters(KERNEL_NAME), convParams));
    knl.getInput(Conv2DWrapKernel.IFMAP_NAME) <== addStreamFromCPU("ifmap");
    knl.getInput(Conv2DWrapKernel.COEFF_NAME) <== addStreamFromCPU("coeff");
    addStreamToCPU("ofmap") <== knl.getOutput(Conv2DWrapKernel.OFMAP_NAME);
  }
  
  // public EngineInterface interfaceDefault(ConvLayerParameters convParams) {
  //   EngineInterface ei = new EngineInterface();
  //   
  //   Conv2DKernelTest.TestData data = new Conv2DKernelTest.TestData(convParams);
  //   ei.setTicks(KERNEL_NAME, data.getNumCycles());
  //   ei.setStream(
  //       "ifmap",
  //       CPUTypes.UINT8,
  //       CPUTypes.UINT8.sizeInBytes() * data.getNumCycles() * data.getIfmapWidth());
  //   ei.setStream(
  //       "coeff",
  //       CPUTypes.UINT8,
  //       CPUTypes.UINT8.sizeInBytes() * data.getNumCycles() * data.getCoeffWidth());
  //   ei.setStream(
  //       "ofmap",
  //       CPUTypes.UINT8,
  //       CPUTypes.UINT8.sizeInBytes() * data.getNumCycles() * data.getOfmapWidth());
  //   
  //   return ei;
  // }
  
  public static void main(String[] args) {
    Conv2DEngineParameters params = new Conv2DEngineParameters(args);
    ConvLayerParameters convParams =
      new ConvLayerParameters.Builder(H, W, C, F, params.getK())
		    .PC(params.getPC())
		    .PF(params.getPF())
		    .PK(params.getPK())
		    .BW(params.getBitWidth())
		    .build();
    Conv2DManager mgr = new Conv2DManager(params, convParams);
    // mgr.createSlicInterface(mgr.interfaceDefault(convParams));
    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    // buildConfig.setMPPRCostTableSearchRange(1, 2);
    // buildConfig.setMPPRParallelism(2);
    mgr.build();
  }

}
