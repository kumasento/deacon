package systolic_matmul;

import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class SystolicMatMulKernel extends Kernel {

  public static final String[] INPUTS = { "x", "y" };
  public static final String[] OUTPUTS = { "z" };

  public SystolicMatMulKernel(KernelParameters params) {
    super(params);

    DFEType T = dfeInt(8);
    
    DFEVar x = io.input(INPUTS[0], T);
    DFEVar y = io.input(INPUTS[1], T);

    io.output(OUTPUTS[0], T).connect(x * y);
  }

}
