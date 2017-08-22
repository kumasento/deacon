package com.custom_computing_ic.maxdeep.kernel.norm;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public abstract class BaseNormKernel extends KernelComponent {

  protected final ConvLayerParameters cp;
  protected final DFEType             T;

  protected final DFEVector<DFEVar>   ifmap;
  protected final DFEVector<DFEVar>   ofmap;

  public BaseNormKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T) {
    super(owner);

    this.cp = cp;

    this.T = T;

    ifmap = getIfmapVecT().newInstance(owner);
    ofmap = getOfmapVecT().newInstance(owner);
  }

  public void setIfmap(DFEVector<DFEVar> ifmap) {
    this.ifmap.connect(ifmap);
  }

  public DFEVector<DFEVar> getOfmap() {
    return ofmap;
  }

  public int getIfmapVecSize() {
    return cp.PF * cp.PK;
  }

  public DFEVectorType<DFEVar> getIfmapVecT() {
    return new DFEVectorType<DFEVar>(T, getIfmapVecSize());
  }

  public int getOfmapVecSize() {
    return cp.PF * cp.PK;
  }

  public DFEVectorType<DFEVar> getOfmapVecT() {
    return new DFEVectorType<DFEVar>(T, getOfmapVecSize());
  }
}
