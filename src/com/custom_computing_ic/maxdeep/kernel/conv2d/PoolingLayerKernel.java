package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Pooling;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.KernelMath;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;

public class PoolingLayerKernel extends ConvLayerKernel {
  private DFEVector<DFEVar> ifmap, ofmap;

  public PoolingLayerKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT) {
    super(owner, cp, T, WT);

    if (cp.inputs.size() > 1 || cp.outputs.size() > 1)
      throw new IllegalArgumentException("Cannot have more than 1 input or output.");
    this.ifmap = ifmapList.get(0);
    this.ofmap = ofmapList.get(0);

    if (cp.PC.get(0) != cp.PF.get(0))
      throw new IllegalArgumentException("PC should equal to PF.");
    if (cp.C != cp.F)
      throw new IllegalArgumentException("C should equal to F.");
    if (cp.pooling == Pooling.AVG)
      throw new IllegalArgumentException("Average pooling is not supported.");
  }

  @Override
  public void initCounterChain(DFEType countT) {
    CounterChain chain = getOwner().control.count.makeCounterChain();
    if (cp.C / cp.PC.get(0) == 1)
      c = constant.var(0).cast(countT);
    else
      c = chain.addCounter(cp.C / cp.PC.get(0), 1).cast(countT);

    h = chain.addCounter(H / PH, 1).cast(countT);
    w = chain.addCounter(W / PW, 1).cast(countT);
    f = constant.var(0).cast(countT);
  }

  /**
   * TODO: rename this function.
   */
  @Override
  public void kernelBody() {
    /* padded input */
    if (cp.dbg)
      debug.simPrintf("ifmap = %KObj%\n", ifmap);
    DFEVector<DFEVar> input =
        control.mux(isInPaddedArea(h, w), ifmap, constant.vect(ifmapList.get(0).getSize(), T, 0));

    /* line buffer */
    lbuf = new ConvLayerLineBuffer(getOwner(), cp, T, 0);
    lbuf.setInput(input);
    DFEVector<DFEVar> kern = lbuf.getOutputVec();

    /* calculate the output */
    for (int pc = 0; pc < cp.PC.get(0); ++pc) {
      int offset = pc * cp.K * cp.K;
      DFEVar out = null;
      if (cp.K == 2) {
        out = KernelMath.max(KernelMath.max(kern.get(offset), kern.get(offset + 1)),
            KernelMath.max(kern.get(offset + 2), kern.get(offset + 3)));
      } else if (cp.K == 3) {
        out = KernelMath.max(
            KernelMath.max(KernelMath.max(KernelMath.max(kern.get(offset), kern.get(offset + 1)),
                               KernelMath.max(kern.get(offset + 2), kern.get(offset + 3))),
                KernelMath.max(KernelMath.max(kern.get(offset + 4), kern.get(offset + 5)),
                    KernelMath.max(kern.get(offset + 6), kern.get(offset + 7)))),
            kern.get(offset + 8));
      } else {
        throw new IllegalArgumentException(String.format("Kernel shape not supported: %d", cp.K));
      }

      ofmap.get(pc).connect(out);
    }
  }

  @Override
  public DFEVar getOfmapEn() {
    return getOfmapBufferWriteEn();
  }
}
