package com.custom_computing_ic.maxdeep.kernel.conv2d;

import java.util.ArrayList;
import java.util.List;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.utils.MathUtils;

/**
 * We try to evaluate the hardware design of doing point-wise convolution on
 * FPGA.
 * 
 * @author rz3515
 * 
 */
public class PointwiseConvolutionKernel extends BaseConvLayerKernel {

  private final DFEVar f, c, h, w;
  private final DFEVector<DFEVar> coeff;

  public PointwiseConvolutionKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T) {
    super(owner, cp, T);

    // if (cp.seq != ConvLayerParameters.CompSeq.FILTER_MAJOR)
    // throw new IllegalArgumentException("Only support filter major in Pointwise
    // convolution.");
    if (cp.STRIDE != 1 && cp.STRIDE != 2)
      throw new IllegalArgumentException("Stride should be 1 or 2.");

    owner.getManager().logMsg("Building pointwise convolution:");
    owner.getManager().logMsg(
        "H = %d W = %d F = %d C = %d PF = %d PC = %d", cp.H, cp.W, cp.F, cp.C, cp.PF, cp.PC);
    owner.getManager().logMsg("Seq = %s", cp.seq.name());

    // initialise counters
    CounterChain chain = control.count.makeCounterChain();
    DFEType countT = dfeInt(32);

    switch (cp.seq) {
      case CHANNEL_MAJOR:
        if (cp.C / cp.PC == 1)
          c = constant.var(0).cast(countT);
        else
          c = chain.addCounter(cp.C / cp.PC, 1).cast(countT);

        if (cp.F / cp.PF == 1)
          f = constant.var(0).cast(countT);
        else
          f = chain.addCounter(cp.F / cp.PF, 1).cast(countT);

        h = chain.addCounter(cp.H / cp.PH, 1).cast(countT);
        w = chain.addCounter(cp.W / cp.PW, 1).cast(countT);
        break;

      case FILTER_MAJOR:
        if (cp.F / cp.PF == 1)
          f = constant.var(0).cast(countT);
        else
          f = chain.addCounter(cp.F / cp.PF, 1).cast(countT);

        if (cp.C / cp.PC == 1)
          c = constant.var(0).cast(countT);
        else
          c = chain.addCounter(cp.C / cp.PC, 1).cast(countT);

        h = chain.addCounter(cp.H / cp.PH, 1).cast(countT);
        w = chain.addCounter(cp.W / cp.PW, 1).cast(countT);
        break;

      default:
        throw new IllegalArgumentException(String.format(
            "Computation sequence %s has not been supported yet", cp.seq));
    }
    // f = chain.addCounter(cp.F / cp.PF, 1);
    // c = chain.addCounter(cp.C / cp.PC, 1);
    // h = (cp.H == cp.PH) ? constant.var(0) : chain.addCounter(cp.H / cp.PH, 1);
    // w = (cp.W == cp.PW) ? constant.var(0) : chain.addCounter(cp.W / cp.PW, 1);

    // Initialize coeff
    if (!cp.coeffOnChip)
      this.coeff = coeffList.get(0);
    else {
      List<Memory<DFEVar>> coeffFMemList = buildCoeffFMemList(T);

      DFEVar addr = getCoeffFMemAddr(dfeUInt(MathUtils.bitsToAddress(getCoeffFMemSize(T))));
      if (cp.dbg)
        debug.simPrintf("coeff FMem addr = %KObj%\n", addr);
      this.coeff = readCoeffFMemList(addr, coeffFMemList, T);
    }

    if (cp.dbg) {
      if (cp.seq == CompSeq.FILTER_MAJOR)
        debug.simPrintf("f = %KObj% c = %KObj% h = %KObj% w = %KObj%\n", f, c, h, w);
      else
        debug.simPrintf("c = %KObj% f = %KObj% h = %KObj% w = %KObj%\n", c, f, h, w);
      debug.simPrintf("ifmap[%3d, %3d, %3d] = %KObj%\n", c, h, w, ifmap);
      debug.simPrintf("coeff[%3d, %3d] = %KObj%\n", f, c, coeff);
      debug.simPrintf("ofmap[%3d, %3d, %3d] = %KObj%\n", f, h, w, ofmap);
    }

    // input feature map buffer
    ConvLayerIfmapBuffer ibuf = new ConvLayerIfmapBuffer(owner, cp, T);
    DFEVar ibufAddr = getIbufAddr(cp, ibuf.getAddrT(), c, h, w);
    DFEVar ibufWriteEn = getIbufWriteEn(f);
    DFEVector<DFEVar> ibufOutput = ibuf.port(ifmap, ibufAddr, ibufWriteEn);

    // output feature map buffer
    ConvLayerOfmapBuffer obuf = new ConvLayerOfmapBuffer(owner, cp, T);
    obuf.setReset(getObufReset(c));

    // dot-product units
    DFEVector<DFEVar> procResult = process(owner, ibufOutput, coeff, cp.PH, cp.PW, cp.PC, cp.PF, T, c, cp.dbg);

    if (cp.dbg) {
      debug.simPrintf("dp_out[%3d, %3d, %3d] = %KObj%\n", f, h, w, procResult);
    }

    // output feature map buffer
    ofmap.connect(obuf.port(procResult, getObufAddr(cp, obuf.getAddrT(), h, w, f),
        getObufWriteEn(cp, h, w)));
  }

  public DFEVar getCoeffFMemAddr(DFEType addrT) {
    return (f.cast(addrT) * constant.var(((int) Math.ceil((double) cp.C / cp.PC))).cast(addrT) + c.cast(addrT))
        .cast(addrT);
  }

  @Override
  public int getIfmapVecSize() {
    return cp.PH * cp.PW * cp.PC;
  }

  @Override
  public DFEVectorType<DFEVar> getIfmapVecT() {
    return ifmapVecT;
  }

  @Override
  public DFEVar getIfmapEn() {
    return f.eq(0);
  }

  @Override
  public List<DFEVar> getCoeffEnList() {
    List<DFEVar> coeffEnList = new ArrayList<DFEVar>();
    coeffEnList.add(h.eq(0) & w.eq(0));

    return coeffEnList;
  }

  @Override
  public List<Integer> getCoeffVecSizeList() {
    List<Integer> coeffVecSizeList = new ArrayList<Integer>();
    coeffVecSizeList.add(cp.PC * cp.PF);
    return coeffVecSizeList;
  }

  @Override
  public DFEVar getOfmapEn() {
    return c.eq(cp.C / cp.PC - 1).and(getObufWriteEn(cp, h, w));
  }

  @Override
  public int getOfmapVecSize() {
    return cp.PF * cp.PH * cp.PW;
  }

  public static DFEVector<DFEVar> process(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      DFEVector<DFEVar> weights, int parHeight, int parWidth, int parInDepth, int parOutDepth,
      DFEType T, DFEVar c, boolean dbg) {
    List<DFEVector<DFEVar>> ifmapPE = getIfmapPE(owner, ifmap, parHeight, parWidth, parInDepth, T);
    List<DFEVector<DFEVar>> weightsPE = getWeightsPE(owner, weights, parInDepth, parOutDepth, T);

    DFEVectorType<DFEVar> resT = new DFEVectorType<DFEVar>(T, parHeight * parWidth * parOutDepth);
    DFEVector<DFEVar> result = resT.newInstance(owner);
    for (int ph = 0; ph < parHeight; ph++) {
      for (int pw = 0; pw < parWidth; pw++) {
        for (int pf = 0; pf < parOutDepth; pf++) {
          DFEVector<DFEVar> currIfmap = ifmapPE.get(ph * parWidth + pw);
          DFEVector<DFEVar> currWeights = weightsPE.get(pf);
          if (dbg) {
            owner.debug.simPrintf("[Pointwise] ifmapPE = %KObj%\n", currIfmap);
            owner.debug.simPrintf("[Pointwise] weightsPE = %KObj%\n", currWeights);
          }

          // in total we instantiate parWidth * parOutDepth * parInDepth number of
          // multipliers
          DotProductKernel dp = new DotProductKernel(owner, parInDepth, T);
          dp.setInputs(currIfmap, currWeights);
          DFEVar out = dp.getOutput();

          if (dbg) {
            owner.debug.simPrintf("[Pointwise] out = %KObj%\n", out);
          }

          result[pf * parWidth * parHeight + ph * parWidth + pw].connect(out);
        }
      }
    }

    return result;
  }

  private DFEVar getIbufAddr(ConvLayerParameters cp, DFEType addrT, DFEVar c, DFEVar h, DFEVar w) {

    // if (cp.seq != CompSeq.FILTER_MAJOR)
    // throw new IllegalArgumentException("cp.seq should be FILTER_MAJOR.");
    DFEVar HEIGHT = constant.var(cp.H / cp.PH).cast(addrT);
    DFEVar WIDTH = constant.var(cp.W / cp.PW).cast(addrT);

    DFEVar addr = (cp.seq == CompSeq.FILTER_MAJOR) ? (c.cast(addrT) * HEIGHT * WIDTH) : constant.var(0).cast(addrT);
    addr += h.cast(addrT) * WIDTH;
    addr += w.cast(addrT);

    return addr;
  }

  private DFEVar getIbufWriteEn(DFEVar f) {
    return f.eq(0);
  }

  private DFEVar getObufReset(DFEVar c) {
    return c.eq(0);
  }

  private DFEVar getObufAddr(ConvLayerParameters cp, DFEType addrT, DFEVar h, DFEVar w, DFEVar f) {
    DFEVar HEIGHT = constant.var(cp.H / cp.PH / cp.STRIDE).cast(addrT);
    DFEVar WIDTH = constant.var(cp.W / cp.PW / cp.STRIDE).cast(addrT);
    DFEVar base = (cp.seq == CompSeq.CHANNEL_MAJOR) ? HEIGHT.mul(WIDTH.mul(f.cast(addrT)))
        : constant.var(0).cast(addrT);
    DFEVar offset = (h.shiftRight(cp.STRIDE - 1).cast(addrT).mul(WIDTH).add(w.shiftRight(cp.STRIDE - 1).cast(addrT)));
    DFEVar addr = base.add(offset);

    if (cp.dbg) {
      debug.simPrintf("[PointwiseConvolution][ObufAddr] HEIGHT = %KObj% WIDTH = %KObj% / %KObj% + %KObj% = %KObj%\n",
          HEIGHT, WIDTH, base, offset, addr);
    }

    return addr;
  }

  private DFEVar getObufWriteEn(ConvLayerParameters cp, DFEVar h, DFEVar w) {
    if (cp.STRIDE == 1)
      return constant.var(1).cast(dfeBool());

    return h.and(constant.var(1).cast(h.getType())).eq(0).and(w.and(constant.var(1).cast(w.getType())).eq(0));
  }

  public static List<DFEVector<DFEVar>> getIfmapPE(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      int parHeight, int parWidth, int parInDepth, DFEType T) {
    List<DFEVector<DFEVar>> splits = new ArrayList<DFEVector<DFEVar>>();

    DFEVectorType<DFEVar> sT = new DFEVectorType<DFEVar>(T, parInDepth);
    for (int ph = 0; ph < parHeight; ph++) {
      for (int pw = 0; pw < parWidth; pw++) {
        DFEVector<DFEVar> split = sT.newInstance(owner);

        for (int pc = 0; pc < parInDepth; pc++) {
          split[pc].connect(ifmap[pc * parWidth * parHeight + ph * parWidth + pw]);
        }

        splits.add(split);
      }
    }

    return splits;
  }

  public static List<DFEVector<DFEVar>> getWeightsPE(KernelBase<?> owner,
      DFEVector<DFEVar> weights, int parInDepth, int parOutDepth, DFEType T) {
    List<DFEVector<DFEVar>> splits = new ArrayList<DFEVector<DFEVar>>();

    DFEVectorType<DFEVar> sT = new DFEVectorType<DFEVar>(T, parInDepth);
    for (int pf = 0; pf < parOutDepth; pf++) {
      DFEVector<DFEVar> split = sT.newInstance(owner);

      for (int pc = 0; pc < parInDepth; pc++) {
        split[pc].connect(weights[pf * parInDepth + pc]);
      }

      splits.add(split);
    }

    return splits;
  }

}
