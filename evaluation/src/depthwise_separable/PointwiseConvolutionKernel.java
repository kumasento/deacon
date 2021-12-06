package depthwise_separable;

import java.util.ArrayList;
import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.custom_computing_ic.maxdeep.lib.stream.BaseStream;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * We try to evaluate the hardware design of doing point-wise convolution on FPGA.
 * 
 * @author rz3515
 * 
 */
public class PointwiseConvolutionKernel extends Kernel {

  public static final String[] INPUTS = {"ifmap", "weights", "bias"};
  public static final String[] OUTPUTS = {"ofmap"};

  public PointwiseConvolutionKernel(KernelParameters params, int tileHeight, int tileWidth,
      int tileInDepth, int tileOutDepth, int parWidth, int parInDepth, int parOutDepth, DFEType T,
      boolean dbg) {
    super(params);

    // initialise counters
    CounterChain chain = control.count.makeCounterChain();
    DFEVar f = chain.addCounter(tileOutDepth / parOutDepth, 1);
    DFEVar c = chain.addCounter(tileInDepth / parInDepth, 1);
    DFEVar h = chain.addCounter(tileHeight, 1);
    DFEVar w =
        (tileWidth == parWidth) ? constant.var(0) : chain.addCounter(tileWidth / parWidth, 1);

    if (dbg) {
      debug.simPrintf("f = %KObj% c = %KObj% h = %KObj% w = %KObj%\n", f, c, h, w);
    }

    // construct convolution layer parameters
    ConvLayerParameters cp =
        (new ConvLayerParameters.Builder(tileHeight, tileWidth, tileInDepth, tileOutDepth, 1))
            .PC(parInDepth).PF(parOutDepth).PK(parWidth).seq(CompSeq.FILTER_MAJOR).dbg(dbg).build();

    // initialise streams
    DFEVar ifmapEn = f.eq(0);
    BaseStream ifmapStream = new BaseStream(INPUTS[0], cp.PC * cp.PK, T, ifmapEn);
    DFEVector<DFEVar> ifmap = ifmapStream.getPlaceholder(getKernel());
    ifmapStream.setIO(getKernel());
    if (dbg) {
      debug.simPrintf("ifmap[%3d, %3d, %3d] = %KObj%\n", c, h, w, ifmap);
    }

    DFEVar weightsEn = h.eq(0) & w.eq(0);
    BaseStream weightsStream = new BaseStream(INPUTS[1], cp.PC * cp.PF, T, weightsEn);
    DFEVector<DFEVar> weights = weightsStream.getPlaceholder(getKernel());
    weightsStream.setIO(getKernel());
    if (dbg) {
      debug.simPrintf("weights[%3d, %3d] = %KObj%\n", f, c, weights);
    }

    DFEVar biasEn = h.eq(0) & w.eq(0) & c.eq(0);
    BaseStream biasStream = new BaseStream(INPUTS[2], cp.PF, T, biasEn);
    DFEVector<DFEVar> bias = biasStream.getPlaceholder(getKernel());
    biasStream.setIO(getKernel());
    if (dbg) {
      debug.simPrintf("bias[%3d] = %KObj%\n", f, bias);
    }

    DFEVar ofmapEn = (c.eq(cp.C / cp.PC - 1));
    BaseStream ofmapStream = new BaseStream(OUTPUTS[0], cp.PF * cp.PK, T, ofmapEn, false);
    DFEVector<DFEVar> ofmap = ofmapStream.getPlaceholder(getKernel());
    ofmapStream.setIO(getKernel());
    if (dbg) {
      debug.simPrintf("ofmap[%3d, %3d, %3d] = %KObj%\n", f, h, w, ofmap);
    }

    // input feature map buffer
    ConvLayerIfmapBuffer ibuf = new ConvLayerIfmapBuffer(getKernel(), cp, T);
    DFEVar ibufAddr = getIbufAddr(cp, ibuf.getAddrT(), c, h, w);
    DFEVar ibufWriteEn = getIbufWriteEn(f);
    DFEVector<DFEVar> ibufOutput = ibuf.port(ifmap, ibufAddr, ibufWriteEn);

    // output feature map buffer
    ConvLayerOfmapBuffer obuf = new ConvLayerOfmapBuffer(getKernel(), cp, T);
    obuf.setReset(getObufReset(c));

    // dot-product units
    DFEVector<DFEVar> procResult =
        process(this, ibufOutput, weights, bias, parWidth, parInDepth, parOutDepth, T, c);
    if (dbg) {
      debug.simPrintf("dp_out[%3d, %3d, %3d] = %KObj%\n", f, h, w, procResult);
    }

    // output feature map buffer
    ofmap.connect(obuf.port(procResult, getObufAddr(cp, obuf.getAddrT(), h, w),
        getObufWriteEn(cp, h, w)));
  }

  public static DFEVector<DFEVar> process(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      DFEVector<DFEVar> weights, DFEVector<DFEVar> bias, int parWidth, int parInDepth,
      int parOutDepth, DFEType T, DFEVar c) {
    List<DFEVector<DFEVar>> ifmapPE = getIfmapPE(owner, ifmap, parWidth, parInDepth, T);
    List<DFEVector<DFEVar>> weightsPE = getWeightsPE(owner, weights, parInDepth, parOutDepth, T);

    DFEVectorType<DFEVar> resT = new DFEVectorType<DFEVar>(T, parWidth * parOutDepth);
    DFEVector<DFEVar> result = resT.newInstance(owner);
    for (int pk = 0; pk < parWidth; pk++) {
      for (int pf = 0; pf < parOutDepth; pf++) {
        DFEVector<DFEVar> currIfmap = ifmapPE.get(pk);
        DFEVector<DFEVar> currWeights = weightsPE.get(pf);

        // in total we instantiate parWidth * parOutDepth * parInDepth number of multipliers
        DotProductKernel dp = new DotProductKernel(owner, parInDepth, T);
        dp.setInputs(currIfmap, currWeights);
        DFEVar out = dp.getOutput();
        DFEVar res =
            (bias != null) ? (out + (c.eq(0) ? bias[pf] : owner.constant.var(0).cast(T))) : out;

        result[pf * parWidth + pk].connect(res);
      }
    }

    return result;
  }

  private DFEVar getIbufAddr(ConvLayerParameters cp, DFEType addrT, DFEVar c, DFEVar h, DFEVar w) {
    DFEVar HEIGHT = constant.var(cp.H).cast(addrT);
    DFEVar WIDTH = constant.var(cp.W / cp.PK).cast(addrT);

    DFEVar addr = c.cast(addrT) * HEIGHT * WIDTH;
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

  private DFEVar getObufAddr(ConvLayerParameters cp, DFEType addrT, DFEVar h, DFEVar w) {
    DFEVar WIDTH = constant.var(cp.W / cp.PK).cast(addrT);

    return (h.cast(addrT) * WIDTH + w.cast(addrT));
  }

  private DFEVar getObufWriteEn(ConvLayerParameters cp, DFEVar h, DFEVar w) {
    return constant.var(1).cast(dfeBool());
  }

  public static List<DFEVector<DFEVar>> getIfmapPE(KernelBase<?> owner, DFEVector<DFEVar> ifmap,
      int parWidth, int parInDepth, DFEType T) {
    List<DFEVector<DFEVar>> splits = new ArrayList<DFEVector<DFEVar>>();

    DFEVectorType<DFEVar> sT = new DFEVectorType<DFEVar>(T, parInDepth);
    for (int pk = 0; pk < parWidth; pk++) {
      DFEVector<DFEVar> split = sT.newInstance(owner);

      for (int pc = 0; pc < parInDepth; pc++) {
        split[pc].connect(ifmap[pc * parWidth + pk]);
      }

      splits.add(split);
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
