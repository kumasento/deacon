package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.fuse.FusedConvLayerParameters;
import com.google.protobuf.WireFormat;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Mem.RamWriteMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.VectorMemory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.Bits;
import com.maxeler.maxcompiler.v2.utils.MathUtils;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public abstract class BaseConvLayerKernel extends KernelComponent {
  protected final ConvLayerParameters cp;
  protected final List<ConvLayerParameters> cps;
  protected final DFEType T, WT;
  protected DFEVector<DFEVar> ifmap, ofmap;
  protected List<DFEVector<DFEVar>> coeffList;
  protected DFEVectorType<DFEVar> ifmapVecT, ofmapVecT;
  protected List<DFEVectorType<DFEVar>> coeffVecTList;
  protected List<Memory<DFEVar>> coeffFMemList;
  protected DFEVector<DFEVar> coeff;
  protected DFEVar initCoeff;

  public BaseConvLayerKernel(
      KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T, boolean useIfmapBuffer) {
    this(owner, cps, T, T, useIfmapBuffer);
  }

  public BaseConvLayerKernel(KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T,
      DFEType WT, boolean useIfmapBuffer) {
    super(owner);

    this.initCoeff = dfeBool().newInstance(owner);
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
      coeffVecTList.add(new DFEVectorType<DFEVar>(WT, coeffVecSizeList.get(i)));
      coeffList.add(coeffVecTList.get(i).newInstance(owner));
    }
  }

  public BaseConvLayerKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT) {
    this(owner, new ArrayList<ConvLayerParameters>(Arrays.asList(cp)), T, WT);
  }

  public BaseConvLayerKernel(
      KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT, boolean useIfmapBuffer) {
    this(owner, new ArrayList<ConvLayerParameters>(Arrays.asList(cp)), T, WT, useIfmapBuffer);
  }

  public BaseConvLayerKernel(
      KernelBase<?> owner, List<ConvLayerParameters> cps, DFEType T, DFEType WT) {
    this(owner, cps, T, WT, true);
  }

  public BaseConvLayerKernel(
      KernelBase<?> owner, FusedConvLayerParameters fcp, DFEType T, DFEType WT) {
    this(owner, fcp.cps, T, WT);
  }

  /* ------------------- Coeff on chip ------------------------------- */

  public int getCoeffFMemSize(DFEType T) {
    return ((int) Math.ceil((double) cp.C / cp.PC)) * ((int) Math.ceil((double) cp.F / cp.PF));
  }

  public abstract DFEVar getCoeffFMemAddr(DFEType addrT);

  public List<Memory<DFEVar>> buildCoeffFMemList(DFEType T) {
    return buildCoeffFMemList(T, true);
  }

  public List<Memory<DFEVar>> buildCoeffFMemList(DFEType T, boolean mapToCPU) {
    List<Memory<DFEVar>> coeffFMemList = new ArrayList<Memory<DFEVar>>();

    for (int pf = 0; pf < cp.PF; ++pf)
      for (int pc = 0; pc < cp.PC; ++pc)
        for (int k = 0; k < cp.K * cp.K; ++k) {
          Memory<DFEVar> memory = mem.alloc(T, getCoeffFMemSize(T));
          if (mapToCPU) {
            String name = String.format("%s_coeff_f%d_c%d_k%d", cp.name, pf, pc, k);
            memory.mapToCPU(name);
            getOwner().getManager().logMsg("Created new memory for coeff: %s", name);
          }

          coeffFMemList.add(memory);
        }

    return coeffFMemList;
  }

  public DFEVector<DFEVar> readCoeffFMemList(DFEVar addr, DFEType T) {
    return readCoeffFMemList(addr, coeffFMemList, T);
  }

  public DFEVector<DFEVar> readCoeffFMemList(
      DFEVar addr, List<Memory<DFEVar>> coeffFMemList, DFEType T) {
    int vecSize = coeffFMemList.size();
    if (vecSize < 1)
      throw new IllegalArgumentException(
          String.format("coeffFMemList should have at least one element, got: %d.", vecSize));

    DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(T, vecSize);
    DFEVector<DFEVar> vec = vecT.newInstance(this.getOwner());

    for (int i = 0; i < vecSize; ++i) {
      DFEVar value = coeffFMemList.get(i).read(addr);
      vec.get(i).connect(value);
    }

    return vec;
  }

  public void readCoeff(DFEVar addr, DFEType T) {
    int vecSize = coeffFMemList.size();
    if (vecSize < 1)
      throw new IllegalArgumentException(
          String.format("coeffFMemList should have at least one element, got: %d.", vecSize));

    for (int i = 0; i < vecSize; ++i) {
      DFEVar value = coeffFMemList.get(i).read(addr);
      this.coeff.get(i).connect(value);
    }
  }

  /**
   * Initialize coeffient through an external interface.
   */
  public void initCoeff(DFEVar initCoeff, DFEVar initCoeffStrm, DFEType T) {
    // Won't do a thing if the initCoeff flag has not turned on.
    if (!cp.initCoeff)
      return;

    this.initCoeff.connect(initCoeff.cast(dfeBool()));
    this.coeffFMemList = new ArrayList<Memory<DFEVar>>();
    getOwner().getManager().logMsg("Initialized coeff FMem list.");

    int memSize = getCoeffFMemSize(T);
    int vecSize = getCoeffVecTList().get(0).getSize();

    CounterChain chain = getOwner().control.count.makeCounterChain();
    DFEVar addr = chain.addCounter(memSize, 1).cast(dfeUInt(MathUtils.bitsToAddress(memSize)));
    DFEVar idx;
    if (vecSize == 1)
      idx = constant.var(0);
    else
      idx = chain.addCounter(vecSize, 1).cast(dfeUInt(MathUtils.bitsToAddress(vecSize)));

    Memory<DFEVector<DFEVar>> memory = mem.alloc(this.coeff.getType(), getCoeffFMemSize(T));

    DFEVector<DFEVar> data = this.coeff.getType().newInstance(getOwner());
    for (int i = 0; i < vecSize; ++i)
      data.get(i).connect(stream.offset(initCoeffStrm, i - vecSize + 1));

    if (cp.dbg) {
      debug.simPrintf("[initCoeff] %KObj% strm = %KObj% addr = %KObj% idx = %KObj% \n",
          this.initCoeff.cast(dfeBool()), initCoeffStrm, addr, idx);
      debug.simPrintf("[initCoeff] data = %KObj%\n", data);
    }

    memory.write(addr, data, idx.eq(vecSize - 1).and(initCoeff));

    this.coeff.connect(memory.read(getCoeffFMemAddr(dfeUInt(MathUtils.bitsToAddress(memSize)))));
    return;
  }

  /** Rom related. */

  public static double[] readROMFile(String key, String fileName) {
    Scanner in = null;

    try {
      in = new Scanner(new File(fileName));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    // TODO: properly throw.

    // Find the correct line.
    String line = in.nextLine();
    while (!(line.startsWith("BEGIN") && line.contains(key))) line = in.nextLine();

    int numElems = in.nextInt();

    double[] rawData = new double[numElems];
    for (int i = 0; i < numElems; ++i) rawData[i] = in.nextDouble();

    return rawData;
  }

  public boolean isTernaryType(DFEType T) {
    return T.getTotalBits() == 2 && T.isUInt();
  }

  public boolean isTernaryType(DFEVectorType<DFEVar> T) {
    return isTernaryType((DFEType) T.getContainedType());
  }

  public double convert(double data, DFEVectorType<DFEVar> T) {
    if (isTernaryType(T))
      return data > 0 ? 1 : (data < 0 ? 3 : 0);
    return data;
  }

  public List<Memory<DFEVar>> getROMList(
      ConvLayerParameters cp, String key, int depth, DFEVectorType<DFEVar> vt) {
    getOwner().getManager().logMsg("Read for key = %s\n", key);
    double[] rawData = readROMFile(key, cp.coeffFile);

    List<Memory<DFEVar>> romList = new ArrayList<Memory<DFEVar>>();
    if (cp.type == Type.DEPTHWISE_SEPARABLE) {
      for (int pc = 0; pc < cp.PC; ++pc) {
        for (int k = 0; k < cp.K * cp.K; ++k) {
          Memory<DFEVar> rom = mem.alloc(vt.getContainedType(), depth);

          double[] part = new double[depth];
          for (int c = 0; c < cp.C; c += cp.PC) {
            double data = rawData[(c + pc) * (cp.K * cp.K) + k];
            part[c / cp.PC] = convert(data, vt);
          }

          rom.setContents(part);
          romList.add(rom);
        }
      }
    } else {
      for (int pf = 0; pf < cp.PF; ++pf) {
        for (int pc = 0; pc < cp.PC; ++pc) {
          for (int k = 0; k < cp.K * cp.K; ++k) {
            Memory<DFEVar> rom = mem.alloc(vt.getContainedType(), depth);

            double[] part = new double[depth];
            for (int f = 0; f < cp.F; f += cp.PF)
              for (int c = 0; c < cp.C; c += cp.PC)
                part[(f / cp.PF) * (cp.C / cp.PC) + (c / cp.PC)] = convert(
                    rawData[(f + pf) * (cp.C * cp.K * cp.K) + (c + pc) * (cp.K * cp.K) + k], vt);

            rom.setContents(part);
            romList.add(rom);
          }
        }
      }
    }

    return romList;
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
      throw new IllegalArgumentException(
          String.format("Coefficient lists are not matching in size: %d (kernel) != %d (parameter)",
              this.coeffList.size(), coeffList.size()));

    for (int i = 0; i < coeffList.size(); i++) this.coeffList.get(i).connect(coeffList.get(i));
  }

  public void setInputs(DFEVector<DFEVar> ifmap, List<DFEVector<DFEVar>> coeffList) {
    this.setIfmap(ifmap);
    this.setCoeffList(coeffList);
  }
}
