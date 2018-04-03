package depthwise_separable;

import java.util.List;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerIfmapBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerOfmapBuffer;
import com.custom_computing_ic.maxdeep.utils.CounterUtils;
import com.custom_computing_ic.maxdeep.utils.StreamUtils;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;

public class DepthwiseSeparableKernel extends Kernel {

  public static final String[] INPUTS = {"ifmap", "depthwise_weights", "pointwise_weights"};
  public static final String[] OUTPUTS = {"ofmap"};

  public DepthwiseSeparableKernel(KernelParameters params, int tileHeight, int tileWidth,
      int tileInDepth, int tileOutDepth, int kernelSize, int stride, int parWidth, int parInDepth,
      int parOutDepth, DFEType T, boolean dbg) {
    super(params);

    /**
     * Constants
     */
    int tileInHeight = tileHeight * stride + kernelSize - 1;
    int tileInWidth = tileWidth * stride + kernelSize - 1;
    if (stride != 1)
      throw new IllegalArgumentException(String.format("stride (%3d) != 1 is not supported yet",
          stride));
    if (tileInWidth % parWidth != 0)
      throw new IllegalArgumentException(String.format(
          "tile input width (%5d) should be divisible by parWidth (%5d)", tileInWidth, parWidth));

    /**
     * Parameters
     */
    ConvLayerParameters cp =
        (new ConvLayerParameters.Builder(tileInHeight, tileInWidth, tileInDepth, tileOutDepth,
            kernelSize)).PK(parWidth).PC(parInDepth).PF(parOutDepth).dbg(dbg).build();
    ConvLayerParameters dcp = cp.createDepthwiseParameters();
    ConvLayerParameters pcp = cp.createPointwiseParameters();

    /**
     * Counters
     */
    DFEType cT = dfeInt(32);
    CounterChain chain = control.count.makeCounterChain();

    int maxOutDepthCount = (int) Math.ceil((double) tileOutDepth / parOutDepth);
    int maxInDepthCount = (int) Math.ceil((double) tileInDepth / parInDepth);
    int maxInWidthCount = (int) Math.ceil((double) tileInWidth / parWidth);

    DFEVar f = CounterUtils.createCounter(chain, maxOutDepthCount, this);
    DFEVar c = CounterUtils.createCounter(chain, maxInDepthCount, this);
    DFEVar ih = chain.addCounter(tileInHeight, stride).cast(cT);
    DFEVar iw = chain.addCounter(maxInWidthCount, stride).cast(cT);
    DFEVar oh = ((ih < kernelSize) ? constant.var(0) : (ih - kernelSize + 1)).cast(cT);
    DFEVar ow =
        (((iw * parWidth < kernelSize - 1) ? constant.var(0) : (iw * parWidth - kernelSize + 1)) / parWidth)
            .cast(cT);
    if (dbg) {
      debug
          .simPrintf("f = %5d c = %5d ih = %5d iw = %5d oh = %5d ow = %5d\n", f, c, ih, iw, oh, ow);
    }

    /**
     * Streams
     */
    // enable signals
    DFEVar ifmapEn = f.eq(0);
    DFEVar depthwiseWeightsEn = f.eq(0) & ih.eq(0) & iw.eq(0);
    DFEVar pointwiseWeightsEn = ih.eq(0) & iw.eq(0);
    DFEVar obufWriteEn = getOBufWriteEn(c, ih, iw, tileHeight, tileWidth, kernelSize, parWidth);
    DFEVar ofmapEn = c.eq(maxInDepthCount - 1) & obufWriteEn;

    // streams objects
    DFEVector<DFEVar> ifmap =
        StreamUtils.createStream(this, INPUTS[0], T, parWidth * parInDepth, ifmapEn, true);
    DFEVector<DFEVar> depthwiseWeights =
        StreamUtils.createStream(this, INPUTS[1], T, parInDepth * kernelSize * kernelSize,
            depthwiseWeightsEn, true);
    DFEVector<DFEVar> pointwiseWeights =
        StreamUtils.createStream(this, INPUTS[2], T, parInDepth * parOutDepth, pointwiseWeightsEn,
            true);
    DFEVector<DFEVar> ofmap =
        StreamUtils.createStream(this, OUTPUTS[0], T, parWidth * parOutDepth, ofmapEn, false);
    if (dbg) {
      debug.simPrintf("ifmap = %KObj%\n", ifmap);
    }

    /**
     * Line buffer:
     * 
     * Convert row-major input stream to size K^2
     */
    ConvLayerLineBuffer lineBuffer = new ConvLayerLineBuffer(getKernel(), dcp, T);
    lineBuffer.setInput(ifmap);
    DFEVector<DFEVar> lbuf = lineBuffer.getOutputVec();
    if (dbg) {
      debug.simPrintf("lbuf  = %KObj%\n", lbuf);
    }

    /**
     * Depthwise Convolution:
     * 
     * Output a DFEVector of size parInDepth * parWidth
     */
    List<DFEVector<DFEVar>> depthwiseIfmapPE =
        DepthwiseConvolutionKernel.getIfmapPE(this, lbuf, parInDepth, parWidth, kernelSize, T);
    List<DFEVector<DFEVar>> depthwiseWeightsPE =
        DepthwiseConvolutionKernel.getWeightsPE(this, depthwiseWeights, parInDepth, parWidth,
            kernelSize, T);
    DFEVector<DFEVar> depthwise =
        DepthwiseConvolutionKernel.process(this, depthwiseIfmapPE, depthwiseWeightsPE, parInDepth,
            parWidth, kernelSize, T);

    if (dbg) {
      debug.simPrintf("depthwise weights   = %KObj%\n", depthwiseWeights);
      for (int i = 0; i < depthwiseIfmapPE.size(); i++) {
        debug.simPrintf("depthwise ifmapPE[%3d]   = %KObj%\n", i, depthwiseIfmapPE[i]);
        debug.simPrintf("depthwise weightsPE[%3d] = %KObj%\n", i, depthwiseWeightsPE[i]);
      }
      debug.simPrintf("depthwise result    = %KObj%\n", depthwise);
    }

    /**
     * Input feature map buffer:
     * 
     * Take depthwise convolution output and store them.
     */
    ConvLayerIfmapBuffer ifmapBuffer = new ConvLayerIfmapBuffer(this, pcp, T);
    DFEVector<DFEVar> ibuf =
        ifmapBuffer.port(depthwise,
            getIBufAddr(c, oh, ow, tileInDepth, tileWidth, parWidth, ifmapBuffer.getAddrT()),
            getIBufWriteEn(f));
    if (dbg) {
      debug.simPrintf("ibuf  = %KObj%\n", ibuf);
    }

    /**
     * Pointwise Convolution Computation
     */
    DFEVector<DFEVar> pointwise =
        PointwiseConvolutionKernel.process(this, ibuf, pointwiseWeights, null, parWidth,
            parInDepth, parOutDepth, T, c);
    if (dbg) {
      debug.simPrintf("pointwiseWeights = %KObj%\n", pointwiseWeights);
      debug.simPrintf("pointwise        = %KObj%\n", pointwise);
    }

    /**
     * Output feature map buffer
     */
    ConvLayerOfmapBuffer ofmapBuffer = new ConvLayerOfmapBuffer(this, pcp, T);
    ofmapBuffer.setReset(c.eq(0));
    ofmap.connect(ofmapBuffer.port(pointwise,
        getOBufAddr(oh, ow, tileWidth, parWidth, ofmapBuffer.getAddrT()), obufWriteEn));
  }

  private DFEVar getOBufWriteEn(DFEVar c, DFEVar ih, DFEVar iw, int tileHeight, int tileWidth,
      int kernelSize, int parWidth) {
    DFEVar boundHeight = ih >= (kernelSize - 1);
    DFEVar boundWidth = iw * parWidth >= (kernelSize - 1);

    return boundHeight & boundWidth;
  }

  private DFEVar getIBufAddr(DFEVar c, DFEVar h, DFEVar w, int tileHeight, int tileWidth,
      int parWidth, DFEType aT) {
    DFEVar HEIGHT = constant.var(tileHeight).cast(aT);
    DFEVar WIDTH = constant.var(tileWidth / parWidth).cast(aT);

    return c.cast(aT) * HEIGHT * WIDTH + h.cast(aT) * WIDTH + w.cast(aT);
  }

  private DFEVar getIBufWriteEn(DFEVar f) {
    return f.eq(0);
  }

  private DFEVar getOBufAddr(DFEVar h, DFEVar w, int tileWidth, int parWidth, DFEType aT) {
    return (h * tileWidth / parWidth + w).cast(aT);
  }
}
