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
  // private DFEVector<DFEVar> ifmap, ofmap;
  // private DFEVector<DFEVar> depthwiseCoeff, pointwiseCoeff;

  public DepthwiseSeparableConvLayerKernel(
      KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT) {
    super(owner, cp, T, WT);
    if (cp.inputs.size() > 1 || cp.outputs.size() > 1)
      throw new IllegalArgumentException("Cannot have more than 1 input or output.");
    // this.ifmap = ifmapList.get(0);
    // this.ofmap = ofmapList.get(0);

    if (!cp.coeffOnChip)
      throw new IllegalArgumentException("coeffOnChip should be true.");
    if (cp.coeffFile.isEmpty())
      throw new IllegalArgumentException("coeffFile should be provided.");

    // create convolution layer parameters for depthwise and pointwise layers
    dcp = cp.createDepthwiseParameters();
    pcp = cp.createPointwiseParameters();
  }

  // @Override
  // public void setIfmap(DFEVector<DFEVar> ifmap, int index) {
  //   if (index != 0)
  //     throw new IllegalArgumentException("Cannot assign index != 0");
  //   this.ifmap.connect(ifmap);
  // }

  // private void setInputs(
  //     DFEVector<DFEVar> ifmap, DFEVector<DFEVar> depthwiseCoeff, DFEVector<DFEVar>
  //     pointwiseCoeff) {
  //   this.ifmap.connect(ifmap);

  //   this.coeffList.get(0).connect(depthwiseCoeff);
  //   this.coeffList.get(1).connect(pointwiseCoeff);
  // }

  // private void initStreams() {
  //   if (cp.type != Type.DEPTHWISE_SEPARABLE)
  //     throw new IllegalArgumentException(
  //         "type of the convolution layer should be DEPTHWISE_SEPARABLE");
  //   // if (cp.seq != CompSeq.FILTER_MAJOR)
  //   //   throw new IllegalArgumentException(
  //   //       "Only FILTER_MAJOR is supported in DEPTHWISE_SEPARABLE type");

  //   /* vector type */
  //   // this.ifmapVecT = new DFEVectorType<DFEVar>(T, getIfmapVecSize());
  //   // this.ofmapVecT = new DFEVectorType<DFEVar>(T, getOfmapVecSize());

  //   /* streams */
  //   // this.ifmap = ifmapVecT.newInstance(getOwner());
  //   // this.ofmap = ofmapVecT.newInstance(getOwner());

  //   // depthwiseCoeff = getDepthwiseCoeffVecT().newInstance(getOwner());
  //   // pointwiseCoeff = getPointwiseCoeffVecT().newInstance(getOwner());

  //   // getOwner().getManager().logMsg("Pointwise coeff type = %s\n", getPointwiseCoeffVecT());

  //   coeffList.clear();
  //   coeffList.add(depthwiseCoeff);
  //   coeffList.add(pointwiseCoeff);
  // }

  public Stream depthwiseConvolution(Stream ifmap) {
    logMsg("Depthwise coeff ROM depth = %d", dcp.getCoeffNumVec(ifmap.index));
    DFEType dwAddrT = dfeUInt(Math.max(
        1, MathUtils.bitsToAddress(MathUtils.nextPowerOfTwo(dcp.getCoeffNumVec(ifmap.index)))));
    DFEVar dwAddr = c.cast(dwAddrT);

    DFEVector<DFEVar> coeff = getROM(dcp, dcp.name, dcp.getCoeffNumVec(ifmap.index),
        dcp.getCoeffVecT(ifmap.index, WT), ifmap.index)
                                  .read(dwAddr);
    Conv2DKernel dwc = new Conv2DKernel(getOwner(), dcp, T, WT, ifmap.index);
    dwc.setInputs(ifmap.data, coeff);
    return new Stream(dwc.getOfmap(), ifmap.index);
  }

  public Stream pointwiseConvolution(Stream ifmap) {
    DFEType pwAddrT = dfeUInt(Math.max(
        1, MathUtils.bitsToAddress(MathUtils.nextPowerOfTwo(pcp.getCoeffNumVec(ifmap.index)))));
    DFEVar pwAddr = f.mul(cp.padC() / cp.PC.get(ifmap.index)).add(c).cast(pwAddrT);
    logMsg("Pointwise coeff ROM depth = %d", pcp.getCoeffNumVec(ifmap.index));

    DFEVector<DFEVar> coeff = getROM(pcp, pcp.name, pcp.getCoeffNumVec(ifmap.index),
        pcp.getCoeffVecT(ifmap.index, WT), ifmap.index)
                                  .read(pwAddr);
    Conv2DKernel pwc = new Conv2DKernel(getOwner(), pcp, T, WT, ifmap.index);
    pwc.setInputs(ifmap.data, coeff);
    return new Stream(pwc.getOfmap(), ifmap.index);
  }

  @Override
  public Stream process(int i) {
    dcp = cp.createDepthwiseParameters();
    pcp = cp.createPointwiseParameters();
    Stream ifmap = padIfmap(new Stream(ifmapList.get(i), i));

    if (needLineBuffer(i, dcp))
      ifmap = lineBuffer(ifmap);
    if (isBypass(i))
      throw new IllegalArgumentException("Cannot bypass.");

    return pointwiseConvolution(bufferizeIfmap(depthwiseConvolution(ifmap)));
  }

  // @Override
  // public void initConvLayer() {
  //   initStreams();

  //   /* padded input */
  //   if (cp.dbg)
  //     debug.simPrintf("ifmap = %KObj%\n", ifmap);
  //   DFEVector<DFEVar> input =
  //       control.mux(isInPaddedArea(h, w), ifmap, constant.vect(ifmapVecT.getSize(), T, 0));

  //   /* line buffer */
  //   lbuf = new ConvLayerLineBuffer(getOwner(), cp, T);
  //   lbuf.setInput(input);
  //   DFEVector<DFEVar> lineBufVec = lbuf.getOutputVec();

  //   /* depthwise conv2d */
  //   DFEVector<DFEVar> depthOfmap = initDepthwiseConv2D(lineBufVec, depthwiseCoeff);

  //   /* ifmap buffer */
  //   ibuf = new ConvLayerIfmapBuffer(
  //       getOwner(), pcp, T, /*loop=*/false, /*forceFull=*/false, /*pad=*/true, "");
  //   DFEVector<DFEVar> ifmapBufVec =
  //       ibuf.port(depthOfmap, getIfmapBufferAddr(ibuf), getIfmapBufferWriteEn());

  //   /* pointwise convolution */
  //   Conv2DKernel pwc = new Conv2DKernel(getOwner(), pcp, T, WT);
  //   pwc.setInputs(ifmapBufVec, pointwiseCoeff);
  //   DFEVector<DFEVar> pointOfmap = pwc.getOfmap();

  //   if (residual != null)
  //     pointOfmap = pointOfmap.add(bufferizeResidual(residual));

  //   /* ofmap buffer */
  //   obuf = new ConvLayerOfmapBuffer(getOwner(), pcp, T, prefix);
  //   obuf.setReset(getOfmapReset());
  //   ofmap.connect(obuf.port(pointOfmap, getOfmapBufferAddr(), getOfmapBufferWriteEn()));
  // }

  // private DFEVector<DFEVar> initDepthwiseConv2D(
  //     DFEVector<DFEVar> lineBufVec, DFEVector<DFEVar> coeff) {
  //   // List<DFEVector<DFEVar>> ifmapVecList = createDepthwiseVecList(lineBufVec);
  //   // List<DFEVector<DFEVar>> coeffVecList = createDepthwiseVecList(depthwiseCoeff);

  //   // DFEVector<DFEVar> depthOfmap = getDepthwiseOfmapVecT().newInstance(getOwner());
  //   Conv2DKernel dwc = new Conv2DKernel(getOwner(), dcp, T, WT);
  //   dwc.setInputs(lineBufVec, coeff);
  //   // for (int i = 0; i < cp.PC; i++) {
  //   //   Conv2DKernel dwc = new Conv2DKernel(getOwner(), dcp, T, WT);
  //   //   dwc.setInputs(ifmapVecList.get(i), coeffVecList.get(i));
  //   //   DFEVector<DFEVar> dwcOfmap = dwc.getOfmap();

  //   //   for (int j = 0; j < cp.PK; j++) {
  //   //     depthOfmap.get(i * cp.PK + j).connect(dwcOfmap.get(j));
  //   //   }
  //   // }

  //   return dwc.getOfmap();
  // }

  // private int getDepthwiseOfmapVecSize(int i) {
  //   return cp.PK * cp.PC.get(i);
  // }

  // private DFEVectorType<DFEVar> getDepthwiseOfmapVecT(int i) {
  //   return new DFEVectorType<DFEVar>(T, getDepthwiseOfmapVecSize(i));
  // }

  // private List<DFEVector<DFEVar>> createDepthwiseVecList(DFEVector<DFEVar> rawVec, int index) {
  //   int vecLen = rawVec.getSize() / cp.PC.get(index);
  //   getOwner().getManager().logMsg(
  //       "vecLen = %d rawVec.getSize() = %d cp.PC = %d\n", vecLen, rawVec.getSize(), cp.PC);
  //   List<DFEVector<DFEVar>> vecList = new ArrayList<DFEVector<DFEVar>>();

  //   for (int i = 0; i < cp.PC.get(iindex); i++) {
  //     DFEVector<DFEVar> vec = (new DFEVectorType<DFEVar>(T, vecLen)).newInstance(getOwner());

  //     for (int j = 0; j < vecLen; j++) vec.get(j).connect(rawVec.get(i * vecLen + j));

  //     vecList.add(vec);
  //   }

  //   return vecList;
  // }

  @Override
  protected DFEVar getIfmapBufferAddr(ConvLayerIfmapBuffer ibuf) {
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

  /**
   * The ifmap buffer in dws is just like an ofmap buffer in a common convolution, without the
   * accumulation logic.
   */
  @Override
  protected DFEVar getIfmapBufferWriteEn() {
    return f.eq(0).and(getOfmapBufferWriteEn());
  }

  public int getDepthwiseCoeffVecSize(int index) {
    return cp.PC.get(index) * dcp.K * dcp.K;
  }

  public DFEVectorType<DFEVar> getDepthwiseCoeffVecT(int index) {
    return new DFEVectorType<DFEVar>(T, getDepthwiseCoeffVecSize(index));
  }

  public DFEVar getDepthwiseCoeffEn() {
    return f.eq(0).and(h.eq(0)).add(w.eq(0));
  }

  public int getPointwiseCoeffVecSize(int index) {
    return pcp.PC.get(index) * pcp.PF.get(index);
  }

  public DFEVectorType<DFEVar> getPointwiseCoeffVecT(int index) {
    return new DFEVectorType<DFEVar>(T, getPointwiseCoeffVecSize(index));
  }

  public DFEVar getPointwiseCoeffEn() {
    return h.eq(0).and(w.eq(0));
  }
}
