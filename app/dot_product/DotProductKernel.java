package dot_product;

import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Stream.OffsetExpr;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;

public class DotProductKernel extends Kernel {

  public DotProductKernel(KernelParameters params) {
    super(params);

    DFEVar x = io.input("x", dfeFloat(8, 24));
    DFEVar y = io.input("y", dfeFloat(8, 24));
    DFEVar z = x.add(y);
    io.output("z", z, dfeFloat(8, 24));
  }
}
