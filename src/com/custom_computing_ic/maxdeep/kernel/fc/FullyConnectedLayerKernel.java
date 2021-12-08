package com.custom_computing_ic.maxdeep.kernel.fc;

import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Mem.RamWriteMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Stream.OffsetExpr;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class FullyConnectedLayerKernel extends Kernel {

  public static final String                  IFMAP_NAME = "ifmap";
  public static final String                  COEFF_NAME = "coeff";
  public static final String                  OFMAP_NAME = "ofmap";

  private final DFEType                       T;

  private final FullyConnectedLayerParameters fp;

  private final DFEVectorType<DFEVar>         ifmapVecT, coeffVecT, ofmapVecT;

  public FullyConnectedLayerKernel(KernelParameters p,
      FullyConnectedLayerParameters fp) {
    super(p);

    if (fp.PC <= 0)
      throw new IllegalArgumentException(String.format(
          "PC (%d) should be larger than 0",
          fp.PC));
    if (fp.W % fp.PC != 0)
      throw new IllegalArgumentException("W % PC should equal 0");
    if (fp.PR <= 0)
      throw new IllegalArgumentException(String.format(
          "PR (%d) should be larger than 0",
          fp.PR));
    if (fp.H % fp.PR != 0)
      throw new IllegalArgumentException("H % PR should equal 0");

    this.T = dfeFix(fp.BW, 0, SignMode.TWOSCOMPLEMENT);
    this.fp = fp;

    OffsetExpr loopLatency =
        stream.makeOffsetAutoLoop(fp.name + "_LOOP_LATENCY");
    DFEVar loopLatencyVal = loopLatency.getDFEVar(getKernel(), dfeUInt(8));

    CounterChain chain = control.count.makeCounterChain();
    DFEVar h = chain.addCounter(fp.H / fp.PR, 1);
    DFEVar w = chain.addCounter(fp.W / fp.PC, 1);
    DFEVar l = chain.addCounter(loopLatencyVal, 1);

    this.ifmapVecT = new DFEVectorType<DFEVar>(T, fp.getIfmapVecSize());
    this.coeffVecT = new DFEVectorType<DFEVar>(T, fp.getCoeffVecSize());
    this.ofmapVecT = new DFEVectorType<DFEVar>(T, fp.getOfmapVecSize());

    DFEVector<DFEVar> ifmap =
        io.input(IFMAP_NAME, ifmapVecT, h.eq(0) & l.eq(0));
    DFEVector<DFEVar> coeff = io.input(COEFF_NAME, coeffVecT, l.eq(0));
    DFEVector<DFEVar> ofmap = ofmapVecT.newInstance(getKernel());

    Memory<DFEVector<DFEVar>> ibuf = mem.alloc(ifmapVecT, fp.W / fp.PC);
    DFEVector<DFEVar> port =
        ibuf.port(w, ifmap, h.eq(0), RamWriteMode.WRITE_FIRST);

    for (int r = 0; r < fp.PR; r++) {
      DotProductKernel dp = new DotProductKernel(getKernel(), fp.PC, T);
      dp.setInputs(port, getCoeffChunkAt(coeff, r));
      DFEVar tmp = dp.getOutput();

      DFEVar carriedSum = T.newInstance(getKernel());
      DFEVar sum = (w.eq(0)) ? constant.var(0).cast(T) : carriedSum;
      DFEVar newSum = sum + tmp;
      carriedSum.connect(stream.offset(newSum, -loopLatency));

      ofmap[r].connect(newSum);
    }

    DFEVar ofmapEn = w.eq(fp.W / fp.PC - 1) & l.eq(loopLatencyVal - 1);

    io.output(OFMAP_NAME, ofmapVecT, ofmapEn).connect(ofmap);
  }

  public DFEVectorType<DFEVar> getIfmapVecT() {
    return ifmapVecT;
  }

  public DFEVectorType<DFEVar> getCoeffVecT() {
    return coeffVecT;
  }

  public DFEVectorType<DFEVar> getOfmapVecT() {
    return ofmapVecT;
  }

  public DFEVector<DFEVar> getCoeffChunkAt(DFEVector<DFEVar> coeff, int r) {
    DFEVectorType<DFEVar> coeffChunkT = new DFEVectorType<DFEVar>(T, fp.PC);
    DFEVector<DFEVar> coeffChunk = coeffChunkT.newInstance(getKernel());
    for (int c = 0; c < fp.PC; c++)
      coeffChunk[c].connect(coeff[r * fp.PC + c]);

    return coeffChunk;
  }
}