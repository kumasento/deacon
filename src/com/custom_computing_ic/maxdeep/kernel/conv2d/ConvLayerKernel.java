/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Output;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.OutputType;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.LineBufferKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.RoundingMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Mem.RamWriteMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.Mux;
import com.maxeler.maxcompiler.v2.utils.MathUtils;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

  protected Map<Integer, ConvLayerIfmapBuffer> ibufMap;

  public class Stream {
    DFEVector<DFEVar> data;
    int index;

    public Stream(DFEVector<DFEVar> data, int index) {
      this.data = data;
      this.index = index;
    }
  }

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

    /** Sanity check */
    if (cp.useWinograd && cp.PAD > 0)
      throw new IllegalArgumentException("Padding should be 0 when using winograd.");
    if (cp.STRIDE != 1 && cp.STRIDE != 2)
      throw new IllegalArgumentException("Stride should be 1 or 2.");

    // check residual connection.
    if (!cp.residual.isEmpty() && cp.seq == CompSeq.FILTER_MAJOR)
      throw new IllegalArgumentException("Cannot use filter major for residual layer");
    if (!cp.residual.isEmpty() && (cp.H != cp.OH || cp.W != cp.OW))
      throw new IllegalArgumentException(
          "The spatial dimensionality shouldn't change for residual connection.");
    if (!cp.residual.isEmpty() && cp.padC() < cp.padF() / cp.PF.get(0) && !shortcut())
      throw new IllegalArgumentException(
          "There will be insufficient input of residuals given cp.padC < cp.padF.");
    // check data type
    if (cp.BW == 1)
      throw new IllegalArgumentException("BW = 1 is not supported.");
    // check ofmap
    if (cp.outputs.get(0).type != OutputType.OFMAP)
      throw new IllegalArgumentException("The first output should be OFMAP type.");
    // for (int i = 1; i < cp.outputs.size(); ++i)
    //   if (cp.outputs.get(i).type == OutputType.OFMAP)
    //     throw new IllegalArgumentException("Cannot have OFMAP output at indices besides 0.");

    this.prefix = prefix;

    /**
     * NOTE: Unlike cp.H, which represents the actual input feature map height,
     * this.H indicates the maximum range of the counter on the H axis. Therefore,
     * it should be padded if cp.PAD is larger than 0.
     */
    this.H =
        cp.useWinograd ? (cp.H + ConvLayerLineBuffer.WINO_LBUF_PADDING_WIDTH) : (cp.H + 2 * cp.PAD);
    this.W =
        cp.useWinograd ? (cp.W + ConvLayerLineBuffer.WINO_LBUF_PADDING_WIDTH) : (cp.W + 2 * cp.PAD);
    this.PH = cp.useWinograd ? ConvLayerLineBuffer.WINO_LBUF_TILE_SIZE : 1;
    this.PW = cp.useWinograd ? ConvLayerLineBuffer.WINO_LBUF_TILE_SIZE : cp.PK;

    owner.getManager().logMsg("Counter H = %d W = %d\n", this.H, this.W);

    this.ibufMap = new HashMap<Integer, ConvLayerIfmapBuffer>();

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

  public boolean shortcut() {
    return cp.inputs.indexOf(cp.residual) != -1;
  }

  public void kernelBody() {
    initConvLayer();
  }

  /**
   * Read from the ifmap buffer by the address for the ofmap buffer.
   * This is useful for duplicating the input for later residual connections.
   * CHANNEL_MAJOR is required to make this work, otherwise, the buffer size will be too small.
   *
   * @return Values read from the ibuf given an address for the imaginary ofmap buffer.
   */
  public DFEVector<DFEVar> getIfmapByOfmapAddr(ConvLayerIfmapBuffer ibuf, int index) {
    if (cp.seq == CompSeq.CHANNEL_MAJOR)
      throw new IllegalArgumentException("Only allowed for filter major.");
    int pad = cp.PAD;

    DFEVar addr = f.mul(H * W).add((oh.add(pad)).mul(W)).add(ow.add(pad)).cast(ibuf.getAddrT());

    DFEVector<DFEVar> read = ibuf.getMem().read(addr);

    debug("[getIfmapByOfmapAddr] At stream %d: addr = %KObj%, data = %KObj%, en = %KObj%\n", index,
        addr, read, getOfmapEn());

    return read;
  }

  public String getCoeffKey(int i) {
    if (i == 0)
      return cp.name;
    return String.format("%s_%d", cp.name, i);
  }

  /**
   * No matter which coeff we're reading, the coeff vector size and type won't change.
   * @param i
   * @return
   */
  public DFEVector<DFEVar> initAndReadCoeff(int i, ConvLayerParameters cp) {
    int depth = cp.getCoeffNumVec(i);
    DFEVar coeffAddr = getCoeffFMemAddr(dfeUInt(MathUtils.bitsToAddress(depth)));
    debug("[initAndReadCoeff] stream %d: addr (" + coeffAddr.getType() + ") = %KObj%\n", i,
        coeffAddr);
    return getROM(cp, getCoeffKey(i), depth, cp.getCoeffVecT(i, WT), i).read(coeffAddr);
  }

  /**
   * Get padded ifmap. Choose between ifmap and zeros by whether we're in the padded area.
   * @param ifmap
   * @return
   */
  @SuppressWarnings("unchecked")
  public Stream padIfmap(Stream ifmap) {
    DFEVectorType<DFEVar> ifmapVecT = ifmap.data.getType();
    DFEVector<DFEVar> zeros = ifmapVecT.newInstance(getOwner());
    for (int i = 0; i < ifmapVecT.getSize(); ++i) zeros.get(i).connect(constant.var(0).cast(T));

    return new Stream(control.mux(isInPaddedArea(h, w), ifmap.data, zeros), ifmap.index);
  }

  /**
   * Check if we should bypass the ifmap at index i.
   * @param i
   * @return
   */
  public boolean isBypass(int i) {
    if (i == 0)
      return false;
    for (Output output : cp.outputs) {
      if (output.type == OutputType.IFMAP && output.index == i)
        return true;
    }
    return false;
  }

  public Stream bufferizeIfmap(Stream ifmap) {
    return bufferizeIfmap(ifmap, cp);
  }

  /**
   * Put ifmap into the ifmap buffer.
   * @param ifmap
   * @return
   */
  public Stream bufferizeIfmap(Stream ifmap, ConvLayerParameters cp) {
    // If it is a bypass ifmap, i.e., no computation will be carried out, we will store all its
    // input.
    boolean forceFull = isBypass(ifmap.index);
    boolean pad = true;

    ConvLayerIfmapBuffer ibuf = new ConvLayerIfmapBuffer(getOwner(), cp, T, /*loop=*/false,
        forceFull, pad, ifmap.index, cp.name + "_" + Integer.toString(ifmap.index));
    DFEVar ifmapBufferWriteAddr = getIfmapBufferAddr(ibuf, c);
    DFEVar ifmapBufferWriteEn = getIfmapBufferWriteEn();
    DFEVar ifmapBufferReadAddr;
    DFEVar mux;
    if (cp.K == 1 && ifmap.index > 0 && this.cp.K == 3) { // shortcut
      ifmapBufferReadAddr = oh.add(cp.PAD).mul(W).add(ow.add(cp.PAD)).cast(ibuf.getAddrT());
      mux = constant.var(false);
    } else {
      ifmapBufferReadAddr = getIfmapBufferAddr(ibuf, c);
      mux = ifmapBufferWriteEn;
    }
    ibufMap.put(ifmap.index, ibuf);

    debug("[bufferizeIfmap] At stream %d: data = %KObj%, write addr ("
            + ifmapBufferWriteAddr.getType() + ") = %KObj% read addr = %KObj% en = %KObj%\n",
        ifmap.index, ifmap.data, ifmapBufferWriteAddr, ifmapBufferReadAddr, ifmapBufferWriteEn);

    return new Stream(
        ibuf.port(ifmap.data, ifmapBufferWriteAddr, ifmapBufferReadAddr, ifmapBufferWriteEn, mux),
        ifmap.index);
  }

  public Stream padAndBufferize(Stream ifmap) {
    if (!isBypass(ifmap.index))
      ifmap = padIfmap(ifmap);
    return bufferizeIfmap(ifmap, cp);
  }

  public Stream padAndBufferize(Stream ifmap, ConvLayerParameters cp) {
    if (!isBypass(ifmap.index))
      ifmap = padIfmap(ifmap);
    return bufferizeIfmap(ifmap, cp);
  }

  public boolean needLineBuffer(int i, ConvLayerParameters cp) {
    return cp.K > 1 && !isBypass(i);
  }

  public Stream lineBuffer(Stream ifmap) {
    return lineBuffer(ifmap, cp);
  }

  public Stream lineBuffer(Stream ifmap, ConvLayerParameters cp) {
    lbuf = new ConvLayerLineBuffer(getOwner(), cp, T, ifmap.index);
    lbuf.setInput(ifmap.data);
    return new Stream(lbuf.getOutputVec(), ifmap.index);
  }

  public Stream convolution(Stream ifmap) {
    return convolution(ifmap, cp);
  }

  public Stream convolution(Stream ifmap, ConvLayerParameters cp) {
    DFEVector<DFEVar> coeff = initAndReadCoeff(ifmap.index, cp);
    Conv2DKernel conv2d = new Conv2DKernel(getOwner(), cp, T, WT, ifmap.index);
    conv2d.setInputs(ifmap.data, coeff);
    return new Stream(conv2d.getOfmap(), ifmap.index);
  }

  /**
   * Process the i-th input.
   * @param i ifmap ID.
   * @return
   */
  public Stream process(int i) {
    ConvLayerParameters cp = i == 0 ? this.cp : this.cp.createShortcutParameters(i);

    Stream ifmap = padAndBufferize(new Stream(ifmapList.get(i), i), cp);
    debug("[padAndBufferize] At stream %d = %KObj%\n", ifmap.index, ifmap.data);

    if (needLineBuffer(i, cp))
      ifmap = lineBuffer(ifmap, cp);
    if (isBypass(i)) {
      return ifmap; // no need for further processing.
    }

    Stream ofmap = convolution(ifmap, cp);

    return ofmap;
  }

  /**
   * Add the residual values from an input port.
   *
   * @param input
   * @param residual
   * @return
   */
  @SuppressWarnings("unchecked")
  public DFEVector<DFEVar> bufferizeResidual(DFEVector<DFEVar> residual) {
    int depth = MathUtils.nextPowerOfTwo(cp.OH * cp.OW);
    logMsg("Residual buffer depth = %d\n", depth);

    Memory<DFEVector<DFEVar>> rbuf = mem.alloc(residual.getType(), depth);
    if (cp.STRIDE != 1)
      throw new IllegalArgumentException("Stride should be 1");
    if (cp.PK != 1)
      throw new IllegalArgumentException("PK should be 1");

    // Write based on the ifmap indices.
    DFEVar writeAddr = (h.sub(cp.PAD))
                           .mul(cp.W / cp.PK)
                           .add(w.sub(cp.PAD))
                           .cast(dfeUInt(MathUtils.bitsToAddress(depth)));
    DFEVar writeEn = getResidualBufferWriteEn();

    // Read based on the ofmap indices
    DFEVar readAddr = oh.mul(cp.OW).add(ow).cast(dfeUInt(MathUtils.bitsToAddress(depth)));

    DFEVector<DFEVar> rbufPort;
    if (cp.K == 1) {
      // In this case read and write addr are the same
      rbufPort = rbuf.port(readAddr, residual, writeEn, RamWriteMode.WRITE_FIRST);
    } else {
      rbuf.write(writeAddr, residual, writeEn);
      rbufPort = rbuf.read(readAddr);
    }
    if (cp.seq != CompSeq.CHANNEL_MAJOR)
      throw new IllegalArgumentException("Should be filter major.");

    DFEVector<DFEVar> residualAdd =
        control.mux(c.eq(f), constant.vect(residual.getSize(), T, 0), rbufPort);
    debug("[Residual] write addr    = %KObj%\n", writeAddr);
    debug("[Residual] read addr     = %KObj%\n", readAddr);
    debug("[Residual] data          = %KObj%\n", residual);
    debug("[Residual] writeEn       = %KObj%\n", writeEn);
    debug("[Residual] to add        = %KObj%\n", residualAdd);
    return residualAdd;
  }

  /**
   * Depending on whether the residual name is in the ifmap list or not.
   *
   * If it is in, then the residual is a result of convolution and should be processed.
   * If not, then the residual directly comes from the external world and should be processed with a
   * buffer.
   *
   * @param residualName
   * @return
   */
  public DFEVector<DFEVar> getResidual(String residualName) {
    int index = cp.inputs.indexOf(residualName);
    if (index >= 0)
      return process(index).data;
    return bufferizeResidual(residual);
  }

  /**
   * Get the convolution result, with residual added if required.
   * @return
   */
  public DFEVector<DFEVar> getResultOfmap() {
    DFEVector<DFEVar> ofmap = process(0).data;
    if (cp.residual.isEmpty())
      return ofmap;
    return ofmap.add(getResidual(cp.residual));
  }

  /**
   * Accumulate in the ofmap buffer.
   * @param ofmap
   * @return
   */
  public DFEVector<DFEVar> bufferizeOfmap(DFEVector<DFEVar> ofmap) {
    obuf = new ConvLayerOfmapBuffer(getOwner(), cp, T, 0, prefix);
    obuf.setReset(getOfmapReset());
    return obuf.port(ofmap, getOfmapBufferAddr(), getOfmapBufferWriteEn());
  }

  public void initConvLayer() {
    /** Prepare the first output. */
    DFEVector<DFEVar> ofmap = bufferizeOfmap(getResultOfmap());
    this.ofmapList.get(0).connect(ofmap);

    /** Connect the rest of the output. */
    for (int i = 1; i < cp.outputs.size(); ++i) {
      if (cp.outputs.get(i).type == OutputType.IFMAP) {
        int index = cp.outputs.get(i).index;
        if (!ibufMap.containsKey(index))
          process(index);

        this.ofmapList.get(i).connect(getIfmapByOfmapAddr(ibufMap.get(index), index));
      } else if (cp.outputs.get(i).type == OutputType.OFMAP) {
        this.ofmapList.get(i).connect(ofmap);
      } else {
        throw new IllegalArgumentException();
      }
    }
  }

  /**
   * The residual buffer can be writen if the input is valid, i.e., it should be the same as
   * IfmapEn.
   * @return
   */
  public DFEVar getResidualBufferWriteEn() {
    return f.eq(0).and(isInPaddedArea(h, w).complement());
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
        return c.eq(cp.padC() / cp.PC.get(0) - 1).and(getOfmapBufferWriteEn());
      case FILTER_MAJOR:
        return c.eq(cp.padC() / cp.PC.get(0) - 1).and(getOfmapBufferWriteEn());
      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }

  public DFEVar getOfmapBufferWriteEn() {
    if (cp.useWinograd) {
      return h.gte(ConvLayerLineBuffer.WINO_LBUF_HEIGHT - 1)
          .and(w.gte(ConvLayerLineBuffer.WINO_LBUF_HEIGHT - 1));
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
          return f.mul(cp.OH * cp.OW / (M * M)).add(oh.mul(cp.OW / M)).add(ow).cast(addrT);

        case FILTER_MAJOR:
          return oh.mul(cp.OW / M).add(ow).cast(addrT);

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

  /**
   * Stays the same for all convolution in the kernel.
   */
  @Override
  public DFEVar getCoeffFMemAddr(DFEType addrT) {
    // TODO: support cases that the weights are in channel major.
    return f.cast(addrT)
        .mul(constant.var(((int) Math.ceil((double) cp.padC() / cp.PC.get(0)))).cast(addrT))
        .add(c.cast(addrT))
        .cast(addrT);
  }

  @Override
  public int getCoeffFMemSize(DFEType T) {
    return ((int) Math.ceil((double) cp.padC() / cp.PC.get(0)))
        * ((int) Math.ceil((double) cp.padF() / cp.PF.get(0)));
  }

  @Override
  public DFEVar getIfmapEn() {
    return f.eq(0).and(isInPaddedArea(h, w).complement());
  }

  @Override
  public List<DFEVar> getCoeffEnList() {
    List<DFEVar> coeffEnList = new ArrayList<DFEVar>();
    DFEVar coeffEn;

    switch (cp.seq) {
      case CHANNEL_MAJOR:
        coeffEn = (h.eq(0)).and(w.eq(0));
        break;
      case FILTER_MAJOR:
        coeffEn = (h.eq(0)).and(w.eq(0));
        break;
      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }

    coeffEnList.add(coeffEn);
    return coeffEnList;
  }

  @Override
  public int getIfmapVecSize(int i) {
    return cp.getIfmapVecSize(i);
  }

  @Override
  public List<Integer> getCoeffVecSizeList() {
    List<Integer> coeffVecSizeList = new ArrayList<Integer>();
    coeffVecSizeList.add(cp.getCoeffVecSize(0));

    return coeffVecSizeList;
  }

  @Override
  public int getOfmapVecSize(int i) {
    return cp.getOfmapVecSize(i);
  }

  public DFEType getCountT() {
    return dfeInt(32);
  }

  public void initCounterChain(DFEType countT) {
    CounterChain chain = getOwner().control.count.makeCounterChain();

    int C = cp.padC();
    int F = cp.padF();
    int PC = cp.PC.get(0);
    int PF = cp.PF.get(0);

    switch (cp.seq) {
      case CHANNEL_MAJOR:
        if (C / PC == 1)
          c = constant.var(0).cast(countT);
        else
          c = chain.addCounter(C / PC, 1).cast(countT);

        if (F / PF == 1)
          f = constant.var(0).cast(countT);
        else
          f = chain.addCounter(F / PF, 1).cast(countT);

        h = chain.addCounter(H / PH, 1).cast(countT);
        w = chain.addCounter(W / PW, 1).cast(countT);
        break;

      case FILTER_MAJOR:
        if (F / PF == 1)
          f = constant.var(0).cast(countT);
        else
          f = chain.addCounter(F / PF, 1).cast(countT);

        if (C / PC == 1)
          c = constant.var(0).cast(countT);
        else
          c = chain.addCounter(C / PC, 1).cast(countT);

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

    oh = control.mux(h.lte(lbufHeight - 1), h.sub(lbufHeight - 1), constant.var(0));
    oh = oh.cast(countT);

    if (cp.useWinograd) {
      ow = control.mux(w.lte(lbufHeight - 1), w.sub(lbufHeight - 1), constant.var(0));
    } else {
      if (cp.PK == 1)
        ow = control.mux(w.lt(cp.K - 1), w.add(1 - cp.K), constant.var(0));
      else
        ow = control.mux(
            w.mul(cp.PK).lt(cp.K - 1), w.mul(cp.PK).add(1 - cp.K).div(cp.PK), constant.var(0));
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
      debug.simPrintf("ofmap buffer addr = %KObj%\n", oh.mul(cp.OW).div(cp.PK).add(ow));
    }
  }

  protected DFEVar getIfmapBufferAddr(ConvLayerIfmapBuffer ibuf, DFEVar c) {
    DFEVar addr;
    switch (cp.seq) {
      case CHANNEL_MAJOR:
        addr = h.mul(W / PW).add(w);
        return addr.cast(ibuf.getAddrT());

      case FILTER_MAJOR:
        addr = c.mul((H / PH) * (W / PW)).add(h.mul((W / PW))).add(w);
        return addr.cast(ibuf.getAddrT());

        // case PIXEL_MAJOR:
        //   addr = c * (H / PH) * (W / PW) + h * (W / PW) + w;
        //   return addr.cast(ibuf.getAddrT());

      default:
        throw new IllegalArgumentException(
            String.format("Computation sequence %s has not been supported yet", cp.seq));
    }
  }

  protected DFEVar getIfmapBufferAddr(ConvLayerIfmapBuffer ibuf) {
    return getIfmapBufferAddr(ibuf, c);
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

  public void debug(String msg, Object... args) {
    if (!cp.dbg)
      return;
    Object[] formatArgs = new Object[args.length + 4];
    formatArgs[0] = f;
    formatArgs[1] = c;
    formatArgs[2] = h;
    formatArgs[3] = w;
    for (int i = 0; i < args.length; ++i) formatArgs[i + 4] = args[i];

    debug.simPrintf("[" + cp.name + "][%KObj%/%KObj%/%KObj%/%KObj%] " + msg, formatArgs);
  }

  public void logMsg(String msg, Object... args) {
    getOwner().getManager().logMsg(msg, args);
  }
}
