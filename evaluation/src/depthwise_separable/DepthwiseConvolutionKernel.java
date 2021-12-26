package depthwise_separable;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import java.util.ArrayList;
import java.util.List;

public class DepthwiseConvolutionKernel extends Kernel {
  public static final String[] INPUTS = {"ifmap", "weights", "bias"};
  public static final String[] OUTPUTS = {"ofmap"};

  public DepthwiseConvolutionKernel(KernelParameters params, int tileHeight, int tileWidth,
      int tileDepth, int kernelSize, int parWidth, int parDepth, DFEType T, boolean dbg) {
    super(params);

    int tileInputHeight = tileHeight + kernelSize - 1;
    int tileInputWidth = tileWidth + kernelSize - 1;

    if (tileInputWidth % parWidth != 0)
      throw new IllegalArgumentException(String.format("The input width of tile (%d) be divisible "
              + "by the number of parallelised units in the width axis (%d)",
          tileInputWidth, parWidth));

    // counters
    DFEType cT = dfeInt(32);
    CounterChain chain = control.count.makeCounterChain();
    // DFEVar c = chain.addCounter(tileDepth / parDepth, 1).cast(cT);
    DFEVar h = chain.addCounter(tileInputHeight, 1).cast(cT);
    DFEVar w = chain.addCounter(tileInputWidth / parWidth, 1).cast(cT);

    if (dbg) {
      debug.simPrintf("h = %KObj% w = %KObj%\n", h, w);
    }

    // initialise the convolution layer parameter
    ConvLayerParameters cp =
        (new ConvLayerParameters.Builder(tileInputHeight, tileInputWidth, tileDepth, 1, kernelSize))
            .PC(parDepth)
            .PK(parWidth)
            .dbg(dbg)
            .build();

    // vector size
    int ifmapVecSize = parWidth * parDepth;
    int weightsVecSize = parDepth * kernelSize * kernelSize;
    int biasVecSize = parDepth;
    int ofmapVecSize = parWidth * parDepth;

    // create streams
    DFEVar ONE = constant.var(1).cast(dfeBool());

    DFEVar ifmapEn = ONE;
    DFEVar weightsEn = h.eq(0) & w.eq(0);
    DFEVar biasEn = h.eq(0) & w.eq(0);
    DFEVar ofmapEn = getOfmapEn(kernelSize, parWidth, h, w);

    DFEVector<DFEVar> ifmap = createStream(INPUTS[0], T, ifmapVecSize, ifmapEn, true);
    DFEVector<DFEVar> weights = createStream(INPUTS[1], T, weightsVecSize, weightsEn, true);
    DFEVector<DFEVar> bias = createStream(INPUTS[2], T, biasVecSize, biasEn, true);
    DFEVector<DFEVar> ofmap = createStream(OUTPUTS[0], T, ofmapVecSize, ofmapEn, false);

    // initialise line buffer
    ConvLayerLineBuffer lineBuffer = new ConvLayerLineBuffer(getKernel(), cp, T, 0);
    lineBuffer.setInput(ifmap);
    DFEVector<DFEVar> lbuf = lineBuffer.getOutputVec();

    // prepare input for the PE
    List<DFEVector<DFEVar>> ifmapPE = getIfmapPE(this, lbuf, parDepth, parWidth, kernelSize, T);
    List<DFEVector<DFEVar>> weightsPE =
        getWeightsPE(this, weights, parDepth, parWidth, kernelSize, T);

    // run PE
    DFEVector<DFEVar> ofmapPE =
        process(this, ifmapPE, weightsPE, parDepth, parWidth, kernelSize, T);

    // add bias
    DFEVector<DFEVar> ofmapBias = addBias(ofmapPE, bias, parDepth, parWidth);

    // connect output
    ofmap.connect(ofmapBias);
  }

  public static DFEVector<DFEVar> process(KernelBase<?> owner, List<DFEVector<DFEVar>> ifmapPE,
      List<DFEVector<DFEVar>> weightsPE, int parDepth, int parWidth, int kernelSize, DFEType T) {
    DFEVectorType<DFEVar> vT = new DFEVectorType<DFEVar>(T, parDepth * parWidth);
    DFEVector<DFEVar> v = vT.newInstance(owner);

    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        DotProductKernel dp = new DotProductKernel(owner, kernelSize * kernelSize, T);
        dp.setInputs(ifmapPE.get(pd * parWidth + pw), weightsPE.get(pd * parWidth + pw));
        v[pd * parWidth + pw].connect(dp.getOutput());
      }
    }

    return v;
  }

  private DFEVector<DFEVar> addBias(
      DFEVector<DFEVar> ofmapPE, DFEVector<DFEVar> bias, int parDepth, int parWidth) {
    DFEVector<DFEVar> ofmapPEBias = ofmapPE.getType().newInstance(getKernel());
    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        ofmapPEBias[pd * parWidth + pw].connect(ofmapPE[pd * parWidth + pw] + bias[pd]);
      }
    }
    return ofmapPEBias;
  }

  private DFEVar getOfmapEn(int kernelSize, int parWidth, DFEVar h, DFEVar w) {
    return getObufWriteEn(kernelSize, parWidth, h, w);
  }

  private DFEVar getObufWriteEn(int kernelSize, int parWidth, DFEVar h, DFEVar w) {
    DFEVar rowAbove = h >= (kernelSize - 1);
    DFEVar colRight = (w * parWidth) >= (kernelSize - 1);

    return rowAbove & colRight;
  }

  /**
   * Create the ifmap for PE array from the output of the line buffer.
   *
   * @param lbuf
   * @return
   */
  public static List<DFEVector<DFEVar>> getIfmapPE(KernelBase<?> owner, DFEVector<DFEVar> lbuf,
      int parDepth, int parWidth, int kernelSize, DFEType T) {
    DFEVectorType<DFEVar> vT = new DFEVectorType<DFEVar>(T, kernelSize * kernelSize);
    List<DFEVector<DFEVar>> ifmapPE = new ArrayList<DFEVector<DFEVar>>();

    int lbufWidth = kernelSize + parWidth - 1;
    int lbufHeight = kernelSize;

    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        DFEVector<DFEVar> v = vT.newInstance(owner);

        for (int kh = 0; kh < kernelSize; kh++) {
          for (int kw = 0; kw < kernelSize; kw++) {
            int idx = pd * lbufHeight * lbufWidth + kh * lbufWidth + (kw + pw);
            v[kh * kernelSize + kw].connect(lbuf[idx]);
          }
        }

        ifmapPE.add(v);
      }
    }

    return ifmapPE;
  }

  public static List<DFEVector<DFEVar>> getWeightsPE(KernelBase<?> owner, DFEVector<DFEVar> weights,
      int parDepth, int parWidth, int kernelSize, DFEType T) {
    DFEVectorType<DFEVar> vT = new DFEVectorType<DFEVar>(T, kernelSize * kernelSize);
    List<DFEVector<DFEVar>> weightsPE = new ArrayList<DFEVector<DFEVar>>();

    for (int pd = 0; pd < parDepth; pd++) {
      for (int pw = 0; pw < parWidth; pw++) {
        DFEVector<DFEVar> v = vT.newInstance(owner);
        for (int k = 0; k < kernelSize * kernelSize; k++) {
          int idx = pd * kernelSize * kernelSize + k;
          v[k].connect(weights[idx]);
        }
        weightsPE.add(v);
      }
    }

    return weightsPE;
  }

  private DFEVector<DFEVar> createStream(
      String name, DFEType T, int vecSize, DFEVar enable, boolean isInput) {
    DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(T, vecSize);
    DFEVector<DFEVar> vec = vecT.newInstance(getKernel());

    if (isInput)
      vec.connect(io.input(name, vecT, enable));
    else
      io.output(name, vecT, enable).connect(vec);

    return vec;
  }
}
