package conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelConfiguration;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelConfiguration.OptimizationOptions.DSPMulAddChainBehaviour;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;

/**
 * Kernel for generating Conv2DKernel different parameters
 * 
 * @author rz3515
 * 
 */
public class Conv2DWrapKernel extends Kernel {

  public static final String        IFMAP_NAME = "ifmap";
  public static final String        COEFF_NAME = "coeff";
  public static final String        OFMAP_NAME = "ofmap";

  public Conv2DWrapKernel(KernelParameters params,
      ConvLayerParameters cp) {
    super(params);
    
    // TODO: remove muladd optimisation for now
    KernelConfiguration config = getKernelConfig();
    config.optimization.setDSPMulAddChainBehavior(DSPMulAddChainBehaviour.IGNORE);

    DFEType scalarT = dfeFix(cp.BW / 2, cp.BW / 2, SignMode.TWOSCOMPLEMENT);
    
    Conv2DKernel conv2d = new Conv2DKernel(getKernel(), cp, scalarT);
    
    DFEVector<DFEVar> ifmap = io.input(IFMAP_NAME, conv2d.getIfmapT());
    DFEVector<DFEVar> coeff = io.input(COEFF_NAME, conv2d.getCoeffT());
    
    conv2d.setInputs(ifmap, coeff);
    
    io.output(OFMAP_NAME, conv2d.getOfmapT()) <== conv2d.getOfmap();
  }
}
