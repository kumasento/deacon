/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.RoundingMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.Mux;
import com.maxeler.maxcompiler.v2.utils.MathUtils;
import java.util.ArrayList;
import java.util.List;

/**
 * This kernel implements a full convolution layer.
 *
 * @author Ruizhe Zhao
 *
 */
public class ConvLayerKernel extends BaseConvLayerKernel {
  public final int H, W, PH, PW;
  protected final String prefix;

  /* counters */
  protected DFEVar h;
  protected DFEVar w;
  protected DFEVar c;
  protected DFEVar f;
  protected DFEVar oh, ow;

  protected ConvLayerIfmapBuffer ibuf;
  protected ConvLayerLineBuffer lbuf;
  protected ConvLayerOfmapBuffer obuf;

  public ConvLayerKernel(
      KernelBase<?> owner, ConvLayerParameters convParams, DFEType T, DFEType WT) {
    this(owner, convParams, T, WT, convParams.name);
  }

  public ConvLayerKernel(KernelBase<?> owner, ConvLayerParameters convParams, DFEType T) {
    this(owner, convParams, T, T, convParams.name);
  }

  public ConvLayerKernel(
      KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT, String prefix) {
    super(owner, cp, T, WT);

    owner.getManager().logMsg("WT = %s", WT.toString());
    owner.getManager().logMsg("coeffOnChip = %b\n", cp.coeffOnChip);
    owner.getManager().logMsg(
        "Input height = %d, output height = %d, pad = %d\n", cp.H, cp.W, cp.PAD);

    if (cp.useWinograd && cp.PAD > 0)
      throw new IllegalArgumentException("Padding should be 0 when using winograd.");
    if (cp.STRIDE != 1 && cp.STRIDE != 2)
      throw new IllegalArgumentException("Stride should be 1 or 2.");

    this.prefix = prefix;

    /**
     * NOTE: Unlike cp.H, which represents the actual input feature map height,
     * this.H indicates the maximum range of the counter on the H axis. Therefore,
     * it should be padded if cp.PAD is larger than 0.
     */
    this.H = cp.useWinograd ? (cp.H + ConvLayerLineBuffer.WINO_LBUF_PADDING_WIDTH) : (cp.H + 2 * cp.PAD);
    this.W = cp.useWinograd ? (cp.W + ConvLayerLineBuffer.WINO_LBUF_PADDING_WIDTH) : (cp.W + 2 * cp.PAD);
    this.PH = cp.useWinograd ? ConvLayerLineBuffer.WINO_LBUF_TILE_SIZE : 1;
    this.PW = cp.useWinograd ? ConvLayerLineBuffer.WINO_LBUF_TILE_SIZE : cp.PK;

    owner.getManager().logMsg("Counter H = %d W = %d\n", this.H, this.W);

    initCounters();

    if (cp.dbg) {
      debug.simPrintf("isInPaddedArea = %KObj%\n", isInPaddedArea(h, w));
    }

    // getOwner().optimization.pushRoundingMode(RoundingMode.TRUNCATE);
    // initConvLayer();
    // getOwner().optimization.popRoundingMode();

    // Replace this part for different kernels.
    kernelBody();
  }

  public void kernelBody() {
    initConvLayer();
  }

  public void initConvLayer() {
    if (!cp.coeffOnChip) {
      if (coeffList.size() != 1)
        throw new IllegalArgumentException(
            String.format("There should be only one coefficient vector"
                + " of standard convolutional layer kernel, got %d",
                coeffList.size()));
      this.coeff = coeffList.get(0);
    } else {
      // TODO: improve
      DFEVar addr = getCoeffFMemAddr(dfeUInt(MathUtils.bitsToAddress(getCoeffFMemSize(WT))));
      if (!cp.initCoeff) {
        this.initCoeff.connect(constant.var(0).cast(dfeBool()));
        if (cp.coeffFile.isEmpty()) {
          List<Memory<DFEVar>> coeffFMemList = buildCoeffFMemList(WT, /* mapToCPU= */!cp.initCoeff);
          this.coeff = readCoeffFMemList(addr, coeffFMemList, WT);
        } else {
          this.coeff = readCoeffFMemList(
              addr, getROMList(cp, cp.name, cp.getCoeffNumVec(), cp.getCoeffVecT(WT)), WT);
        }
      } else {
        this.coeff = getCoeffVecTList().get(0).newInstance(getOwner());
      }
    }

    DFEVector<DFEVar> zeroVec = ifmapVecT.newInstance(getOwner());
    for (int i = 0; i < ifmapVecT.getSize(); ++i)
      zeroVec.get(i).connect(constant.var(0).cast(T));

    if (cp.dbg)
      debug.simPrintf("ifmap = %KObj%\n", ifmap);
    DFEVector<DFEVar> input = control.mux(isInPaddedArea(h, w), ifmap, zeroVec);

    /* ifmap buffer */
    ibuf = new ConvLayerIfmapBuffer(getOwner(), cp, T);
    DFEVector<DFEVar> ifmapBufVec = ibuf.port(input, getIfmapBufferAddr(),
        getIfmapBufferWriteEn().and(initCoeff.complement()));

    /* line buffer */
    lbuf = new ConvLayerLineBuffer(getOwner(), cp, T);
    lbuf.setInput(ifmapBufVec);
    DFEVector<DFEVar> lineBufVec = lbuf.getOutputVec();

    /* conv2d */
    DFEVector<DFEVar> conv2dOfmap = null;
    if (cp.BW == 1) {
      BinarizedConv2DKernel conv2d = new BinarizedConv2DKernel(getOwner(), cp);
      conv2d.setInputs(lineBufVec, coeff);
      conv2dOfmap = conv2d.getOfmap();
    } else {
      Conv2DKernel conv2d = new Conv2DKernel(getOwner(), cp, T, WT);
      conv2d.setInputs(lineBufVec, coeff);
      conv2dOfmap = conv2d.getOfmap();
    }

    /* output buffer */
    obuf = new ConvLayerOfmapBuffer(
        getOwner(), cp, conv2dOfmap.getElementsAsList()[0].getType(), prefix);
    obuf.setReset(getOfmapReset());

    if (cp.BW == 1) {
      DFEVector<DFEVar> rawOfmap = obuf.port(conv2dOfmap, getOfmapBufferAddr(), getOfmapBufferWriteEn());
      for (int i = 0; i < rawOfmap.getSize(); i++)
        this.ofmap[i].connect((rawOfmap[i] > 1).cast(T));

    } else {
      // TODO: change 1 here to be a real threshold value
      this.ofmap.connect(obuf.port(
          conv2dOfmap, getOfmapBufferAddr(), getOfmapBufferWriteEn().and(initCoeff.complement())));
    }
  }

  public DFEVar isInPaddedArea(DFEVar h, DFEVar w) {
    DFEVar pad = constant.var(cp.PAD).cast(getCountT());
    DFEVar cnt_max_h = constant.var(this.H).cast(getCountT());
    DFEVar cnt_max_w = constant.var(this.W).cast(getCountT());
    return h.lt(pad).or(h.gte(cnt_max_h.sub(pad))).or(w.lt(pad)).or(w.gte(cnt_max_w.sub(pad)));
  }

  public DFEVar getOfmapReset() {
    switch (cp.seq) {
      case CHANNEL_MAJOR:
        return c.eq(0);
      case FILTER_MAJOR:
        return c.eq(0);
      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }

  @Override
  public DFEVar getOfmapEn() {
    switch (cp.seq) {
      case CHANNEL_MAJOR:
        return c.eq(cp.C / cp.PC - 1) & getOfmapBufferWriteEn();
      case FILTER_MAJOR:
        return c.eq(cp.C / cp.PC - 1) & getOfmapBufferWriteEn();
      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }

  public DFEVar getOfmapBufferWriteEn() {
    if (cp.useWinograd) {
      return (h >= ConvLayerLineBuffer.WINO_LBUF_HEIGHT - 1)
          & (w >= ConvLayerLineBuffer.WINO_LBUF_HEIGHT - 1);
    } else {
      DFEVar cond = h.gte(cp.K - 1).and(w.mul(cp.PK).gte(cp.K - 1));
      if (cp.STRIDE == 1)
        return cond;

      // STRIDE is 2, check output is mod 2 == 0.
      return cond.and(oh.and(constant.var(1).cast(oh.getType())).eq(0))
          .and(ow.and(constant.var(1).cast(ow.getType())).eq(0));
    }
  }

  public DFEVar getOfmapBufferAddr() {
    return getOfmapBufferAddr(obuf.getAddrT());
  }

  public DFEVar getOfmapBufferAddr(DFEType addrT) {
    if (cp.useWinograd) {
      int M = WinogradTransform.M;
      switch (cp.seq) {
        case CHANNEL_MAJOR:
          return (f * (cp.OH * cp.OW / (M * M)) + oh * cp.OW / M + ow).cast(addrT);

        case FILTER_MAJOR:
          return (oh * cp.OW / M + ow).cast(addrT);

        default:
          throw new IllegalArgumentException(
              String.format("Computation sequence %s has not been supported yet", cp.seq));
      }

    } else {
      switch (cp.seq) {
        case CHANNEL_MAJOR:
          return f.mul(cp.OH * cp.OW / cp.PK)
              .add(oh.shiftRight(cp.STRIDE - 1).mul(cp.OW / cp.PK))
              .add(ow.shiftRight(cp.STRIDE - 1))
              .cast(addrT);
        case FILTER_MAJOR:
          return oh.shiftRight(cp.STRIDE - 1)
              .mul(cp.OW / cp.PK)
              .add(ow.shiftRight(cp.STRIDE - 1))
              .cast(addrT);

        default:
          throw new IllegalArgumentException(
              String.format("Computation sequence %s has not been supported yet", cp.seq));
      }
    }
  }

  public DFEVar getCoeffFMemAddr(DFEType addrT) {
    // TODO: support cases that the weights are in channel major.
    return (f.cast(addrT) * constant.var(((int) Math.ceil((double) cp.C / cp.PC))).cast(addrT)
        + c.cast(addrT))
            .cast(addrT);
  }

  @Override
  public DFEVar getIfmapEn() {
    return getIfmapBufferWriteEn().and(isInPaddedArea(h, w).complement());
  }

  @Override
  public List<DFEVar> getCoeffEnList() {
    List<DFEVar> coeffEnList = new ArrayList<DFEVar>();
    DFEVar coeffEn;

    switch (cp.seq) {
      case CHANNEL_MAJOR:
        coeffEn = (h.eq(0)) & (w.eq(0));
        break;
      case FILTER_MAJOR:
        coeffEn = (h.eq(0)) & (w.eq(0));
        break;
      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }

    coeffEnList.add(coeffEn);
    return coeffEnList;
  }

  @Override
  public int getIfmapVecSize() {
    return cp.getIfmapVecSize();
  }

  @Override
  public List<Integer> getCoeffVecSizeList() {
    List<Integer> coeffVecSizeList = new ArrayList<Integer>();
    coeffVecSizeList.add(cp.getCoeffVecSize());

    return coeffVecSizeList;
  }

  @Override
  public int getOfmapVecSize() {
    return cp.getOfmapVecSize();
  }

  public DFEType getCountT() {
    return dfeInt(32);
  }

  public void initCounterChain(DFEType countT) {
    CounterChain chain = getOwner().control.count.makeCounterChain();
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

        h = chain.addCounter(H / PH, 1).cast(countT);
        w = chain.addCounter(W / PW, 1).cast(countT);
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

        h = chain.addCounter(H / PH, 1).cast(countT);
        w = chain.addCounter(W / PW, 1).cast(countT);
        break;

      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }

  public void initCounters() {
    DFEType countT = getCountT();
    initCounterChain(countT);

    // counters for the output fmap
    int lbufHeight = ConvLayerLineBuffer.getLineBufferHeight(cp);

    oh = (h <= lbufHeight - 1) ? constant.var(0) : h - lbufHeight + 1;
    oh = oh.cast(countT);

    if (cp.useWinograd) {
      ow = (w <= lbufHeight - 1) ? constant.var(0) : w - lbufHeight + 1;
    } else {
      ow = (w * cp.PK < (cp.K - cp.K / 2)) ? constant.var(0) : (w * cp.PK + cp.K / 2 - cp.K) / cp.PK;
    }
    ow = ow.cast(countT);

    getOwner().getManager().logMsg("oh is %s", oh);
    getOwner().getManager().logMsg("ow is %s", ow);

    if (cp.dbg) {
      debug.simPrintf("%s: \n", cp.name);
      debug.simPrintf("f = %d c = %d h = %d w = %d\n", f, c, h, w);
      debug.simPrintf("ifmapEn = %d\n", getIfmapEn());
      debug.simPrintf("ofmapEn = %d\n", getOfmapEn());
      debug.simPrintf("oh = %KObj% ow = %KObj%\n", oh, ow);
      debug.simPrintf("ofmap buffer addr = %KObj%\n", oh * cp.OW / cp.PK + ow);
    }
  }

  protected DFEVar getIfmapBufferAddr() {
    DFEVar addr;
    switch (cp.seq) {
      case CHANNEL_MAJOR:
        addr = h * (W / PW) + w;
        return addr.cast(ibuf.getAddrT());

      case FILTER_MAJOR:
        addr = c * (H / PH) * (W / PW) + h * (W / PW) + w;
        return addr.cast(ibuf.getAddrT());

      case PIXEL_MAJOR:
        addr = c * (H / PH) * (W / PW) + h * (W / PW) + w;
        return addr.cast(ibuf.getAddrT());

      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }

  protected DFEVar getIfmapBufferWriteEn() {
    switch (cp.seq) {
      case CHANNEL_MAJOR:
        return f.eq(0);

      case FILTER_MAJOR:
        return f.eq(0);

      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }
}
