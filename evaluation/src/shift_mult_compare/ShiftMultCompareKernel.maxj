package shift_mult_compare;

import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;

public class ShiftMultCompareKernel extends Kernel {

  public static final String[] INPUTS = {"x", "y"};
  public static final String[] OUTPUTS = {"z"};

  public ShiftMultCompareKernel(KernelParameters params) {
    super(params);

    DFEType T = dfeFix(8, 8, SignMode.TWOSCOMPLEMENT);

    optimization.pushDSPFactor(0.0);
    DFEVar x = io.input(INPUTS[0], T);
    DFEVar y = io.input(INPUTS[1], T);
    optimization.popDSPFactor();

    DFEVar z1 = x * y;
    DFEVar z2 = x << 2;

    io.output(OUTPUTS[0], T).connect(z1 + z2);
  }
}
