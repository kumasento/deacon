package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Mem.RamWriteMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Stream.OffsetExpr;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.MathUtils;

/**
 * Buffer for input feature map stream.
 *
 * HOW TO USE:
 *
 * <pre>
 * ConvLayerParameters cp = ...;
 * ConvLayerIfmapBuffer ibuf = new ConvLayerIfmapBuffer(getKernel(), cp,
 * scalarT); dataOut <== ibuf.port(data, addr, writeEn);
 * </pre>
 *
 * @author Ruizhe Zhao
 *
 */
public class ConvLayerIfmapBuffer extends KernelComponent {
  private final ConvLayerParameters cp;
  private int C;

  private final DFEVectorType<DFEVar> portVecT;
  private final DFEType addrT;

  private final Memory<DFEVector<DFEVar>> mem;
  private final DFEVector<DFEVar> port;
  private final DFEVar addr;
  private final DFEVar readAddr;
  private final DFEVector<DFEVar> data;
  private final DFEVar writeEn;
  private final DFEVar mux;

  public ConvLayerIfmapBuffer(KernelBase<?> owner, ConvLayerParameters params, DFEType scalarT) {
    this(owner, params, scalarT, false, false, true, 0, "");
  }

  public ConvLayerIfmapBuffer(KernelBase<?> owner, ConvLayerParameters params, DFEType scalarT,
      boolean forceFull, boolean pad) {
    this(owner, params, scalarT, false, forceFull, pad, 0, "");
  }

  public ConvLayerIfmapBuffer(KernelBase<?> owner, ConvLayerParameters params, DFEType scalarT,
      boolean loop, boolean forceFull, boolean pad, int index, String prefix) {
    super(owner);

    this.cp = params;

    // width depends on the stream parallelism, depth doesn't (stays the same for all ifmap buffer).
    this.C = this.cp.padC();
    int width = getWidth(index);
    int depth = MathUtils.nextPowerOfTwo(getDepth(pad, forceFull));

    owner.getManager().logMsg(String.format("Ifmap buffer configuration %d x %d", depth, width));
    // System.out.printf("[ConvLayerIfmapBuffer] width = %d depth = %d\n",
    // width, depth);

    this.addrT = dfeUInt(MathUtils.bitsToAddress(depth));
    this.portVecT = new DFEVectorType<DFEVar>(scalarT, width);

    this.mem = owner.mem.alloc(portVecT, depth);
    this.addr = addrT.newInstance(owner);
    this.readAddr = addrT.newInstance(owner);
    this.data = portVecT.newInstance(owner);
    this.writeEn = dfeBool().newInstance(owner);
    this.mux = dfeBool().newInstance(owner);

    OffsetExpr writeLatency = stream.makeOffsetAutoLoop(prefix + "_IBUF_WRITE_LATENCY");

    owner.getManager().logMsg(String.format("loop = %s", loop));
    if (!loop) {
      // this.port = mem.port(addr, data, writeEn, RamWriteMode.WRITE_FIRST);
      this.port = control.mux(mux, mem.read(readAddr), data);
      mem.write(addr, data, writeEn);
    } else {
      this.port = mem.read(addr);

      mem.write(stream.offset(addr, -writeLatency), stream.offset(data, -writeLatency),
          stream.offset(writeEn, -writeLatency));
    }

    if (cp.dbg) {
      debug.simPrintf("[ConvLayerIfmapBuffer] input = %KObj%\n", data);
      debug.simPrintf("[ConvLayerIfmapBuffer] output = %KObj%\n", port);
      debug.simPrintf("[ConvLayerIfmapBuffer] addr = %KObj% %KObj%\n", addr,
          stream.offset(addr, -writeLatency));
      debug.simPrintf("[ConvLayerIfmapBuffer] writeEn = %KObj% %KObj%\n", writeEn,
          stream.offset(writeEn, -writeLatency));
    }
  }

  public DFEType getAddrT() {
    return addrT;
  }

  public DFEVector<DFEVar> port(DFEVector<DFEVar> data, DFEVar addr, DFEVar writeEn) {
    this.data.connect(data);
    this.addr.connect(addr);
    this.readAddr.connect(addr);
    this.writeEn.connect(writeEn);
    this.mux.connect(writeEn);
    return this.port;
  }

  public DFEVector<DFEVar> port(
      DFEVector<DFEVar> data, DFEVar addr, DFEVar readAddr, DFEVar writeEn, DFEVar mux) {
    this.data.connect(data);
    this.addr.connect(addr);
    this.readAddr.connect(readAddr);
    this.writeEn.connect(writeEn);
    this.mux.connect(mux);
    return this.port;
  }

  public DFEVectorType<DFEVar> getPortVecT() {
    return portVecT;
  }

  public int getWidth(int index) {
    if (cp.type == Type.POINTWISE)
      return cp.PH * cp.PW * cp.PC.get(index);

    return cp.useWinograd ? cp.PC.get(index) * ConvLayerLineBuffer.WINO_LBUF_NUM_PIPES
                          : cp.PC.get(index) * cp.PK;
  }

  public int getDepth(boolean pad, boolean forceFull) {
    if (cp.type != Type.STANDARD && cp.type != Type.DEPTHWISE_SEPARABLE)
      throw new IllegalArgumentException(
          "Only STANDARD / DEPTHWISE_SEPARABLE is supported for now.");

    int height = cp.H;
    int width = cp.W;
    if (pad) {
      height += 2 * cp.PAD;
      width += 2 * cp.PAD;
    }

    if (cp.seq == CompSeq.FILTER_MAJOR || forceFull)
      return (cp.padC() / cp.PC.get(0)) * height * (width / cp.PK);
    if (cp.seq == CompSeq.CHANNEL_MAJOR)
      return height * (width / cp.PK);

    throw new IllegalArgumentException(
        String.format("Computation sequence %s has not been supported yet", cp.seq));
  }

  public Memory<DFEVector<DFEVar>> getMem() {
    return mem;
  }
}
