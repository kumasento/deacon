package com.custom_computing_ic.maxdeep.kernel.norm;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.MathUtils;

public class BatchNormKernel extends BaseNormKernel {
  public static final String COEFF_ROM_SUFFIX = "batch_norm_coeff";
  public static final String BIAS_ROM_SUFFIX = "batch_norm_bias";

  private final DFEVector<DFEVar> coeff;
  private final DFEVector<DFEVar> bias;

  private final DFEVar coeffROMAddr;
  private final DFEVar biasROMAddr;

  public BatchNormKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType scalarT) {
    super(owner, cp, scalarT);

    coeffROMAddr = getROMAddrT().newInstance(owner);
    biasROMAddr = getROMAddrT().newInstance(owner);

    coeff = initCoeffROM();
    bias = initBiasROM();

    // core computation
    for (int pf = 0; pf < cp.PF.get(0); pf++) {
      for (int pk = 0; pk < cp.PK; pk++) {
        ofmap[pf * cp.PK + pk].connect(coeff[pf] * ifmap[pf * cp.PK + pk] + bias[pf]);
      }
    }
  }

  public void setAddr(DFEVar addr) {
    coeffROMAddr.connect(addr);
    biasROMAddr.connect(addr);
  }

  private int getROMWidth() {
    return cp.PF.get(0);
  }

  private int getROMDepth() {
    return cp.F / cp.PF.get(0);
  }

  public DFEType getROMAddrT() {
    return dfeUInt(MathUtils.bitsToAddress(getROMDepth()));
  }

  public DFEVector<DFEVar> initCoeffROM() {
    Memory<DFEVector<DFEVar>> mem = getOwner().mem.alloc(getPortVecT(), getROMDepth());
    mem.mapToCPU(cp.name + "_" + COEFF_ROM_SUFFIX);
    return mem.read(coeffROMAddr);
  }

  public DFEVector<DFEVar> initBiasROM() {
    Memory<DFEVector<DFEVar>> mem = getOwner().mem.alloc(getPortVecT(), getROMDepth());
    mem.mapToCPU(cp.name + "_" + BIAS_ROM_SUFFIX);
    return mem.read(biasROMAddr);
  }

  public DFEVectorType<DFEVar> getPortVecT() {
    return new DFEVectorType<DFEVar>(T, getROMWidth());
  }
}
