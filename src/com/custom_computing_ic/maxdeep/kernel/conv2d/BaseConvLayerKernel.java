package com.custom_computing_ic.maxdeep.kernel.conv2d;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.fuse.FusedConvLayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;

public abstract class BaseConvLayerKernel extends KernelComponent {

  protected final ConvLayerParameters cp;
  protected final List<ConvLayerParameters> cps;
  protected final DFEType T, WT;
  protected DFEVector<DFEVar> ifmap, ofmap;
  protected List<DFEVector<DFEVar>> coeffList;
  protected DFEVectorType<DFEVar> ifmapVecT, ofmapVecT;
  protected List<DFEVectorType<DFEVar>> coeffVecTList;

  public BaseConvLayerKernel(KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T,
      boolean useIfmapBuffer) {
    this(owner, cps, T, T, useIfmapBuffer);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T, DFEType WT,
      boolean useIfmapBuffer) {
    super(owner);

    this.cps = cps;
    this.cp = cps.get(0); // be compatible with a single layer case

    this.T = T;
    this.WT = WT;

    ifmapVecT = new DFEVectorType<DFEVar>(T, getIfmapVecSize());
    ifmap = ifmapVecT.newInstance(getOwner());

    ofmapVecT = new DFEVectorType<DFEVar>(T, getOfmapVecSize());
    ofmap = ofmapVecT.newInstance(getOwner());

    // create coefficents
    List<Integer> coeffVecSizeList = getCoeffVecSizeList();
    coeffVecTList = new ArrayList<DFEVectorType<DFEVar>>();
    coeffList = new ArrayList<DFEVector<DFEVar>>();
    for (int i = 0; i < coeffVecSizeList.size(); i++) {
      coeffVecTList.add(new DFEVectorType<DFEVar>(T, coeffVecSizeList.get(i)));
      coeffList.add(coeffVecTList.get(i).newInstance(owner));
    }
  }

  public BaseConvLayerKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT) {
    this(owner, new ArrayList<ConvLayerParameters>(Arrays.asList(cp)), T, WT);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT,
      boolean useIfmapBuffer) {
    this(owner, new ArrayList<ConvLayerParameters>(Arrays.asList(cp)), T, WT, useIfmapBuffer);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T, DFEType WT) {
    this(owner, cps, T, WT, true);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, FusedConvLayerParameters fcp, DFEType T, DFEType WT) {
    this(owner, fcp.cps, T, WT);
  }

  /* ------------------- Coeff on chip ------------------------------- */

  public int getCoeffFMemSize(DFEType T) {
    return ((int) Math.ceil((double) cp.C / cp.PC))
        * ((int) Math.ceil((double) cp.F / cp.PF));

  }

  public abstract DFEVar getCoeffFMemAddr(DFEType addrT);

  public List<Memory<DFEVar>> buildCoeffFMemList(DFEType T) {
    List<Memory<DFEVar>> coeffFMemList = new ArrayList<Memory<DFEVar>>();

    for (int pf = 0; pf < cp.PF; ++pf)
      for (int pc = 0; pc < cp.PC; ++pc)
        for (int k = 0; k < cp.K * cp.K; ++k) {
          Memory<DFEVar> memory = mem.alloc(T, getCoeffFMemSize(T));
          String name = String.format("%s_coeff_f%d_c%d_k%d", cp.name, pf, pc, k);
          memory.mapToCPU(name);

          getOwner().getManager().logMsg("Created new memory for coeff: %s", name);

          coeffFMemList.add(memory);
        }

    return coeffFMemList;
  }

  public DFEVector<DFEVar> readCoeffFMemList(DFEVar addr, List<Memory<DFEVar>> coeffFMemList, DFEType T) {
    int vecSize = coeffFMemList.size();
    if (vecSize < 1)
      throw new IllegalArgumentException(String.format("coeffFMemList should have at least one element, got: %d.",
          vecSize));

    DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(T, vecSize);
    DFEVector<DFEVar> vec = vecT.newInstance(this.getOwner());

    for (int i = 0; i < vecSize; ++i) {
      DFEVar value = coeffFMemList.get(i).read(addr);
      vec.get(i).connect(value);
    }

    return vec;
  }

  public DFEVector<DFEVar> getIfmap() {
    return ifmap;
  }

  public DFEVectorType<DFEVar> getIfmapVecT() {
    return ifmapVecT;
  }

  public abstract DFEVar getIfmapEn();

  public abstract int getIfmapVecSize();

  public List<DFEVector<DFEVar>> getCoeffList() {
    return coeffList;
  }

  public List<DFEVectorType<DFEVar>> getCoeffVecTList() {
    return coeffVecTList;
  }

  public abstract List<DFEVar> getCoeffEnList();

  public abstract List<Integer> getCoeffVecSizeList();

  public DFEVectorType<DFEVar> getOfmapVecT() {
    return ofmapVecT;
  }

  public abstract DFEVar getOfmapEn();

  public abstract int getOfmapVecSize();

  public DFEVector<DFEVar> getOfmap() {
    return ofmap;
  }

  public void setIfmap(DFEVector<DFEVar> ifmap) {
    this.ifmap.connect(ifmap);
  }

  public void setCoeffList(List<DFEVector<DFEVar>> coeffList) {

    if (this.coeffList.size() != coeffList.size())
      throw new IllegalArgumentException(String.format(
          "Coefficient lists are not matching in size: %d (kernel) != %d (parameter)",
          this.coeffList.size(), coeffList.size()));

    for (int i = 0; i < coeffList.size(); i++)
      this.coeffList.get(i).connect(coeffList.get(i));
  }

  public void setInputs(DFEVector<DFEVar> ifmap, List<DFEVector<DFEVar>> coeffList) {
    this.setIfmap(ifmap);
    this.setCoeffList(coeffList);
  }
}
