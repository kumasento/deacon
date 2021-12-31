/**
 *
 */
package com.custom_computing_ic.maxdeep.lib;

import com.custom_computing_ic.maxdeep.utils.AdderTree;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.Optimization;
import com.maxeler.maxcompiler.v2.kernelcompiler.op_management.FixOpBitSizeMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.op_management.MathOps;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * A kernel component responsible for creating dot-product logic.
 *
 * The dot-product is implemented by an array of multipliers followed by an
 * adder tree with the same width.
 *
 * @author Ruizhe Zhao
 * @since 18/06/2017
 */
public class DotProductKernel extends KernelComponent {
  private final DFEVector<DFEVar> vecA;
  private final DFEVector<DFEVar> vecB;
  private final DFEVar out;
  private final DFEVectorType<DFEVar> vecAT, vecBT;

  public DotProductKernel(KernelBase<?> owner, int vecSize, DFEType type) {
    this(owner, vecSize, type, type, false);
  }

  public DotProductKernel(KernelBase<?> owner, int vecSize, DFEType type, DFEType type2) {
    this(owner, vecSize, type, type2, false);
  }

  /**
   * Constructor for the DotProductKernelComponent.
   *
   * @param owner   owner design of this component
   * @param vecSize size of the vectors to be computed
   * @param type    data type.s
   */
  public DotProductKernel(
      KernelBase<?> owner, int vecSize, DFEType typeA, DFEType typeB, boolean dbg) {
    super(owner);

    if (vecSize <= 0)
      throw new IllegalArgumentException("vecSize should be larger than 0");

    vecAT = new DFEVectorType<DFEVar>(typeA, vecSize);
    vecBT = new DFEVectorType<DFEVar>(typeB, vecSize);
    DFEVectorType<DFEVar> vecCT =
        new DFEVectorType<DFEVar>((typeA.getTotalBits() == 8 && typeB.getTotalBits() == 27)
                ? dfeFix(45, 0, SignMode.TWOSCOMPLEMENT)
                : typeB,
            vecSize);

    vecA = vecAT.newInstance(owner);
    vecB = vecBT.newInstance(owner);

    DFEVector<DFEVar> vecC = vecCT.newInstance(owner);

    // An array of multipliers will be initialized here.
    if (typeB.isUInt() && typeB.getTotalBits() == 2) {
      vecC = vecAT.newInstance(owner);
      for (int i = 0; i < vecSize; ++i)
        vecC.get(i).connect(control.mux(vecB.get(i), constant.var(0).cast(typeA), vecA.get(i),
            constant.var(0).cast(typeA), vecA.get(i).neg()));

    } else {
      if (typeA.getTotalBits() == 8 && typeB.getTotalBits() == 27) {
        owner.optimization.pushFixOpMode(
            Optimization.bitSizeExact(45), Optimization.offsetExact(0), MathOps.ALL);
        for (int i = 0; i < vecSize; i++)
          vecC.get(i).connect(vecA.get(i).cast(dfeInt(18)).mul(vecB.get(i).cast(dfeInt(27))));
        owner.optimization.popFixOpMode(MathOps.ALL);
      } else {
        for (int i = 0; i < vecSize; i++) vecC.get(i).connect(vecA.get(i).mul(vecB.get(i)));
      }
    }

    if (dbg) {
      debug.simPrintf("[DotProd] vecA = %KObj%\n", vecA);
      debug.simPrintf("[DotProd] vecB = %KObj%\n", vecB);
      debug.simPrintf("[DotProd] vecC = %KObj%\n", vecC);
    }

    // Output is the result of an adder-tree based reduction
    out = AdderTree.reduce(vecC.getElementsAsList());
  }

  public void setInputs(DFEVector<DFEVar> vecA, DFEVector<DFEVar> vecB) {
    this.vecA.connect(vecA);
    this.vecB.connect(vecB);
  }

  public DFEVectorType<DFEVar> getInputVecT() {
    return vecAT;
  }

  public DFEVar getOutput() {
    return out;
  }

  public DFEType getOutputT() {
    return out.getType();
  }
}
