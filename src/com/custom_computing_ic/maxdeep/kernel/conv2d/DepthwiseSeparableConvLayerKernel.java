package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.MathUtils;
import java.util.ArrayList;
import java.util.List;

public class DepthwiseSeparableConvLayerKernel extends ConvLayerKernel {
  private ConvLayerParameters dcp, pcp;
  private DFEVector<DFEVar> depthwiseCoeff, pointwiseCoeff;

  public DepthwiseSeparableConvLayerKernel(
      KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT) {
    super(owner, cp, T, WT);

    if (!cp.coeffOnChip)
      throw new IllegalArgumentException("coeffOnChip should be true.");
    if (cp.coeffFile.isEmpty())
      throw new IllegalArgumentException("coeffFile should be provided.");
  }

  public void setInputs(DFEVector<DFEVar> ifmap) {
    this.ifmap.connect(ifmap);

    DFEType dwAddrT = dfeUInt(MathUtils.bitsToAddress(dcp.getCoeffNumVec()));
    DFEVar dwAddr = c.cast(dwAddrT);
    getOwner().getManager().logMsg("Depthwise coeff ROM depth = %d", dcp.getCoeffNumVec());

    DFEType pwAddrT = dfeUInt(MathUtils.bitsToAddress(pcp.getCoeffNumVec()));
    DFEVar pwAddr = f.mul(cp.C / cp.PC).add(c).cast(pwAddrT);
    getOwner().getManager().logMsg("Pointwise coeff ROM depth = %d", pcp.getCoeffNumVec());

    this.depthwiseCoeff.connect(readCoeffFMemList(
        dwAddr, getROMList(dcp.name, dcp.getCoeffNumVec(), dcp.getCoeffVecT(WT)), WT));
    this.pointwiseCoeff.connect(readCoeffFMemList(
        pwAddr, getROMList(pcp.name, pcp.getCoeffNumVec(), pcp.getCoeffVecT(WT)), WT));
  }

  public void setInputs(
      DFEVector<DFEVar> ifmap, DFEVector<DFEVar> depthwiseCoeff, DFEVector<DFEVar> pointwiseCoeff) {
    this.ifmap.connect(ifmap);

    this.coeffList.get(0).connect(depthwiseCoeff);
    this.coeffList.get(1).connect(pointwiseCoeff);
  }

  public void initStreams() {
    if (cp.type != Type.DEPTHWISE_SEPARABLE)
      throw new IllegalArgumentException(
          "type of the convolution layer should be DEPTHWISE_SEPARABLE");
    // if (cp.seq != CompSeq.FILTER_MAJOR)
    //   throw new IllegalArgumentException(
    //       "Only FILTER_MAJOR is supported in DEPTHWISE_SEPARABLE type");

    // create convolution layer parameters for depthwise and pointwise layers
    dcp = cp.createDepthwiseParameters();
    pcp = cp.createPointwiseParameters();

    /* vector type */
    this.ifmapVecT = new DFEVectorType<DFEVar>(T, getIfmapVecSize());
    this.ofmapVecT = new DFEVectorType<DFEVar>(T, getOfmapVecSize());

    /* streams */
    this.ifmap = ifmapVecT.newInstance(getOwner());
    this.ofmap = ofmapVecT.newInstance(getOwner());

    depthwiseCoeff = getDepthwiseCoeffVecT().newInstance(getOwner());
    pointwiseCoeff = getPointwiseCoeffVecT().newInstance(getOwner());

    getOwner().getManager().logMsg("Pointwise coeff type = %s\n", getPointwiseCoeffVecT());

    coeffList.clear();
    coeffList.add(depthwiseCoeff);
    coeffList.add(pointwiseCoeff);
  }

  @Override
  public void initConvLayer() {
    initStreams();

    /* padded input */
    if (cp.dbg)
      debug.simPrintf("ifmap = %KObj%\n", ifmap);
    DFEVector<DFEVar> input =
        control.mux(isInPaddedArea(h, w), ifmap, constant.vect(ifmapVecT.getSize(), T, 0));

    /* line buffer */
    lbuf = new ConvLayerLineBuffer(getOwner(), cp, T);
    lbuf.setInput(input);
    DFEVector<DFEVar> lineBufVec = lbuf.getOutputVec();

    /* depthwise conv2d */
    DFEVector<DFEVar> depthOfmap = initDepthwiseConv2D(lineBufVec, depthwiseCoeff);

    /* ifmap buffer */
    ibuf = new ConvLayerIfmapBuffer(getOwner(), pcp, T);
    DFEVector<DFEVar> ifmapBufVec =
        ibuf.port(depthOfmap, getIfmapBufferAddr(), getIfmapBufferWriteEn());

    /* pointwise convolution */
    Conv2DKernel pwc = new Conv2DKernel(getOwner(), pcp, T, WT);
    pwc.setInputs(ifmapBufVec, pointwiseCoeff);
    DFEVector<DFEVar> pointOfmap = pwc.getOfmap();

    /* ofmap buffer */
    obuf = new ConvLayerOfmapBuffer(getOwner(), pcp, T, prefix);
    obuf.setReset(getOfmapReset());
    ofmap.connect(obuf.port(pointOfmap, getOfmapBufferAddr(), getOfmapBufferWriteEn()));
  }

  private DFEVector<DFEVar> initDepthwiseConv2D(
      DFEVector<DFEVar> lineBufVec, DFEVector<DFEVar> coeff) {
    List<DFEVector<DFEVar>> ifmapVecList = createDepthwiseVecList(lineBufVec);
    List<DFEVector<DFEVar>> coeffVecList = createDepthwiseVecList(depthwiseCoeff);

    DFEVector<DFEVar> depthOfmap = getDepthwiseOfmapVecT().newInstance(getOwner());
    for (int i = 0; i < cp.PC; i++) {
      Conv2DKernel dwc = new Conv2DKernel(getOwner(), dcp, T, WT);
      dwc.setInputs(ifmapVecList.get(i), coeffVecList.get(i));
      DFEVector<DFEVar> dwcOfmap = dwc.getOfmap();

      for (int j = 0; j < cp.PK; j++) {
        depthOfmap[i * cp.PK + j].connect(dwcOfmap[j]);
      }
    }

    return depthOfmap;
  }

  private int getDepthwiseOfmapVecSize() {
    return cp.PK * cp.PC;
  }

  private DFEVectorType<DFEVar> getDepthwiseOfmapVecT() {
    return new DFEVectorType<DFEVar>(T, getDepthwiseOfmapVecSize());
  }

  private List<DFEVector<DFEVar>> createDepthwiseVecList(DFEVector<DFEVar> rawVec) {
    int vecLen = rawVec.getSize() / cp.PC;
    System.out.printf(
        "vecLen = %d rawVec.getSize() = %d cp.PC = %d\n", vecLen, rawVec.getSize(), cp.PC);
    List<DFEVector<DFEVar>> vecList = new ArrayList<DFEVector<DFEVar>>();

    for (int i = 0; i < cp.PC; i++) {
      DFEVector<DFEVar> vec = (new DFEVectorType<DFEVar>(T, vecLen)).newInstance(getOwner());

      for (int j = 0; j < vecLen; j++) vec[j].connect(rawVec[i * vecLen + j]);

      vecList.add(vec);
    }

    return vecList;
  }

  @Override
  protected DFEVar getIfmapBufferAddr() {
    DFEVar addr;
    switch (pcp.seq) {
      case CHANNEL_MAJOR:
        addr = h.mul(pcp.W / pcp.PW).add(w);
        return addr.cast(ibuf.getAddrT());

      case FILTER_MAJOR:
        addr = c.mul((pcp.H / pcp.PH) * (pcp.W / pcp.PW)).add(h.mul(pcp.W / pcp.PW)).add(w);
        return addr.cast(ibuf.getAddrT());

      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }

  @Override
  public DFEVar getIfmapEn() {
    return f.eq(0).and(isInPaddedArea(h, w).complement());
  }

  /**
   * The ifmap buffer in dws is just like an ofmap buffer in a common convolution, without the
   * accumulation logic.
   */
  @Override
  protected DFEVar getIfmapBufferWriteEn() {
    return f.eq(0).and(getOfmapBufferWriteEn());
  }

  public int getDepthwiseCoeffVecSize() {
    return cp.PC * dcp.K * dcp.K;
  }

  public DFEVectorType<DFEVar> getDepthwiseCoeffVecT() {
    return new DFEVectorType<DFEVar>(T, getDepthwiseCoeffVecSize());
  }

  public DFEVar getDepthwiseCoeffEn() {
    return f.eq(0) & h.eq(0) & w.eq(0);
  }

  public int getPointwiseCoeffVecSize() {
    return pcp.PC * pcp.PF;
  }

  public DFEVectorType<DFEVar> getPointwiseCoeffVecT() {
    return new DFEVectorType<DFEVar>(T, getPointwiseCoeffVecSize());
  }

  public DFEVar getPointwiseCoeffEn() {
    return h.eq(0) & w.eq(0);
  }
}
