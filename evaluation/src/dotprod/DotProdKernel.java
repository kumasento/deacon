package dotprod;

import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Stream.OffsetExpr;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;

/**
 * Kernel for generating DotProd with different parameters, and performing a reduced sum of all
 * dot-product results.
 * 
 * @author rz3515
 * 
 */
public class DotProdKernel extends Kernel {

  public static final String VEC_A = "VEC_A";
  public static final String VEC_B = "VEC_B";
  public static final String RES = "RESULT";
  public static final String NUM_VECS = "NUM_VECS";
  public static final String OFFSET = "offset";

  public DotProdKernel(KernelParameters parameters, int vecSize, int bitWidth, DFEType T) {
    super(parameters);

    DotProductKernel dotProd = new DotProductKernel(this, vecSize, T);

    // total number of vectors
    DFEVar numVecs = io.scalarInput(NUM_VECS, dfeUInt(32));

    // loop offset
    OffsetExpr offset = stream.makeOffsetAutoLoop(OFFSET);
    DFEVar latency = offset.getDFEVar(this, dfeUInt(8));

    // counter
    CounterChain chain = control.count.makeCounterChain();
    DFEVar i = chain.addCounter(numVecs, 1);
    DFEVar j = chain.addCounter(latency, 1);

    // perform computation
    DFEVector<DFEVar> vecA = io.input(VEC_A, dotProd.getInputVecT(), j.eq(0));
    DFEVector<DFEVar> vecB = io.input(VEC_B, dotProd.getInputVecT(), j.eq(0));
    dotProd.setInputs(vecA, vecB);
    DFEVar result = dotProd.getOutput();

    // sum
    DFEVar ZERO = constant.var(T, 0.0);
    DFEVar carriedSum = T.newInstance(this);
    DFEVar sum = i.eq(0) ? ZERO : carriedSum;
    DFEVar newSum = sum + result;
    DFEVar newSumOffset = stream.offset(newSum, -offset);
    carriedSum.connect(newSumOffset);

    // output
    io.output(RES, T, i.eq(numVecs - 1) & j.eq(latency - 1)).connect(newSum);
  }
}
