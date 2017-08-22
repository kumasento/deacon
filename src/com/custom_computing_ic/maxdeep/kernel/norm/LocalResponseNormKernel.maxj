package com.custom_computing_ic.maxdeep.kernel.norm;

import java.util.ArrayList;
import java.util.List;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class LocalResponseNormKernel extends BaseNormKernel {

  // private final DFEVar oh, ow, ih, iw;

  /**
   * LRN Kernel.
   * 
   * This kernel component should be used as an individual kernel appended after
   * the CONV wrap kernel.
   * 
   * @param owner
   * @param cp
   * @param scalarT
   */
  public LocalResponseNormKernel(KernelBase<?> owner, ConvLayerParameters cp,
      DFEType scalarT) {
    super(owner, cp, scalarT);

    if (cp.PK != 1)
      throw new IllegalArgumentException(
          "PK should equal 1 when CONV is appended with LRN");
    if (cp.OH % cp.LRNLocalSize != 0)
      throw new IllegalArgumentException("H % LRNLocalSize should equal 0");
    if (cp.OW % cp.LRNLocalSize != 0)
      throw new IllegalArgumentException("W % LRNLocalSize should equal 0");

    // CounterChain chain = owner.control.count.makeCounterChain();
    // oh = chain.addCounter(cp.OH / cp.LRNLocalSize, 1).cast(dfeInt(32));
    // ow = chain.addCounter(cp.OW / cp.LRNLocalSize, 1).cast(dfeInt(32));
    //
    //
  }

  // public DFEVar getOfmapValid() {
  // return (h > 0 & (h % cp.LRNLocalSize).eq(0))
  // }

  public int getLRNWindowSize() {
    return cp.LRNLocalSize * cp.LRNLocalSize;
  }

  public DFEVectorType<DFEVar> getLRNWindowT() {
    return new DFEVectorType<DFEVar>(T, getLRNWindowSize());
  }

  public List<DFEVector<DFEVar>> createLRNWindowList() {
    List<DFEVector<DFEVar>> LRNWindowsList = new ArrayList<DFEVector<DFEVar>>();

    for (int f = 0; f < cp.PF; f++) {
      DFEVector<DFEVar> LRNWindow = getLRNWindowT().newInstance(getOwner());
      // PK is 1 so we can access ifmap value only by f.
      DFEVar inp = ifmap[f];

      for (int kx = 0; kx < cp.LRNLocalSize; kx++)
        for (int ky = 0; ky < cp.LRNLocalSize; ky++) {
          // LRNWindow[kx * cp.LRNLocalSize + ky].connect();
        }
    }

    return LRNWindowsList;
  }

}
