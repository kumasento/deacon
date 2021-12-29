package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.OutputType;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import java.util.ArrayList;
import java.util.List;

/**
 * Kernel for generating Conv2DKernel different parameters
 *
 * @author rz3515
 *
 */
public class ConvLayerWrapKernel extends Kernel {
  public static final String IFMAP_NAME = "ifmap";
  public static final String COEFF_NAME = "coeff";
  public static final String DEPTHWISE_COEFF_NAME = "depthwise_coeff";
  public static final String POINTWISE_COEFF_NAME = "pointwise_coeff";
  public static final String OFMAP_NAME = "ofmap";
  public static final String RESIDUAL_NAME = "residual";
  public static final String INIT_COEFF_NAME = "init_coeff";
  public static final String INIT_COEFF_STREAM_NAME = "init_coeff_strm";
  public static final String INIT_COEFF_STREAM_OUT_NAME = "init_coeff_strm_out";

  private final ConvLayerParameters cp;
  private final DFEType T;

  public ConvLayerWrapKernel(KernelParameters params, ConvLayerParameters cp) {
    this(params, cp, 1);
  }

  public ConvLayerWrapKernel(
      KernelParameters params, ConvLayerParameters cp, int numCoeffFifoSplits) {
    super(params);

    // KernelConfiguration config = getKernelConfig();
    // config.optimization.setFIFOImplementationBRAMThreshold(2048);
    // config.optimization
    // .setDSPMulAddChainBehavior(DSPMulAddChainBehaviour.IGNORE);

    this.cp = cp;

    // TODO: This type will return a signed integer when BW > 1
    T = (cp.BW == 1) ? dfeUInt(1) : cp.getDFEType();
    DFEType WT = cp.WBW < 8 ? dfeUInt(cp.WBW) : cp.getDFEType(cp.WBW);

    getManager().logMsg("T = %s", T);
    getManager().logMsg("WT = %s", WT);

    // DFEVar initCoeff = constant.var(0).cast(dfeBool());
    // DFEVar initCoeffStrm = constant.var(0).cast(WT);
    // if (cp.initCoeff && cp.type != Type.IDENTITY) {
    //   getManager().logMsg("Init coeffient signal\n");
    //   initCoeff = io.scalarInput(INIT_COEFF_NAME, dfeBool());
    //   initCoeffStrm = io.input(INIT_COEFF_STREAM_NAME, dfeUInt(8), initCoeff).cast(WT);
    //   io.output(INIT_COEFF_STREAM_OUT_NAME, initCoeffStrm.cast(dfeUInt(8)), dfeUInt(8),
    //   initCoeff);
    // }

    if (!cp.coeffOnChip)
      throw new IllegalArgumentException("Only coeffOnChip is supported for now.");
    if (cp.initCoeff)
      throw new IllegalArgumentException("Don't use initCoeff.");

    /**
     * TODO: what is the proper way to initialise conv and cast the type?
     */
    if (cp.type == Type.IDENTITY) {
      DFEVectorType<DFEVar> vecT = new DFEVectorType<DFEVar>(T, cp.getIfmapVecSize(0));
      DFEVector<DFEVar> ifmap = io.input(IFMAP_NAME, vecT);

      for (int i = 0; i < cp.numOutputs; ++i) {
        getManager().logMsg("Connecting to output: %s\n", getOfmapName(i));
        io.output(getOfmapName(i), vecT).connect(ifmap);
      }
    } else if (cp.type == Type.CONCAT) {
      // Just implement the concat logic here.

      // if (cp.PF.get(0) != 1)
      //   throw new IllegalArgumentException("PF should not be set for CONCAT");
      // if (cp.PC.size() > 1 || cp.PF.size() > 1)
      //   throw new IllegalArgumentException();

      if (cp.inputs.size() < 2)
        throw new IllegalArgumentException("CONCAT should have more than 1 input.");

      int PC = cp.PC.get(0);
      for (int i = 1; i < cp.inputs.size(); ++i)
        if (cp.PC.get(i) != PC)
          throw new IllegalArgumentException("PC should be the same for CONCAT.");
      for (int i = 1; i < cp.outputs.size(); ++i)
        if (cp.PF.get(i) != cp.inputs.size() * PC)
          throw new IllegalArgumentException("PF should be num_inputs * PC for CONCAT.");

      List<DFEVector<DFEVar>> ifmapList = new ArrayList<DFEVector<DFEVar>>();
      for (int i = 0; i < cp.inputs.size(); ++i)
        ifmapList.add(
            io.input(getIfmapName(i), new DFEVectorType<DFEVar>(T, cp.getIfmapVecSize(0))));

      DFEVectorType<DFEVar> outT =
          new DFEVectorType<DFEVar>(T, cp.getIfmapVecSize(0) * cp.inputs.size());
      DFEVector<DFEVar> outVec = outT.newInstance(this);

      for (int i = 0; i < cp.inputs.size(); ++i)
        for (int j = 0; j < PC; ++j) outVec.get(i * PC + j).connect(ifmapList.get(i).get(j));

      if (cp.dbg) {
        DFEVar cnt = control.count.simpleCounter(1000);
        debug.simPrintf("[Concat] counter: %d\n", cnt);
        for (int i = 0; i < cp.inputs.size(); ++i)
          debug.simPrintf("[Concat] At %d = %KObj%\n", i, ifmapList.get(i));
        debug.simPrintf("[Concat] Out vec = %KObj%\n", outVec);
      }

      for (int i = 0; i < cp.numOutputs; ++i) io.output(getOfmapName(i), outT).connect(outVec);

    } else if (cp.type == Type.POOLING) {
      if (cp.PC.size() > 1 || cp.PF.size() > 1)
        throw new IllegalArgumentException();

      PoolingLayerKernel pool = new PoolingLayerKernel(getKernel(), cp, T, WT);
      DFEVector<DFEVar> ifmap = io.input(IFMAP_NAME, pool.getIfmapVecT(0), pool.getIfmapEn());
      pool.setIfmap(ifmap, 0);

      for (int i = 0; i < cp.numOutputs; ++i) {
        getManager().logMsg("Connecting to output: %s\n", getOfmapName(i));
        io.output(getOfmapName(i), pool.getOfmapVecT(0), pool.getOfmapEn())
            .connect(pool.getOfmap());
      }

    } else if (cp.type == Type.STANDARD || cp.type == Type.POINTWISE
        || cp.type == Type.DEPTHWISE_SEPARABLE) {
      BaseConvLayerKernel conv = ConvLayerKernelFactory.create(getKernel(), cp, T, WT);

      for (int i = 0; i < cp.inputs.size(); ++i)
        conv.setIfmap(io.input(getIfmapName(i), conv.getIfmapVecT(i), conv.getIfmapEn()), i);

      if (!cp.residual.isEmpty() && cp.inputs.indexOf(cp.residual) == -1) {
        // if (!conv.getIfmapVecT().equals(conv.getOfmapVecT()))
        //   throw new IllegalArgumentException(
        //       String.format("The ifmap vec and ofmap vec don't match: %s vs %s",
        //           conv.getIfmapVecT(), conv.getOfmapVecT()));
        DFEVector<DFEVar> residual =
            io.input(RESIDUAL_NAME, conv.getOfmapVecT(0), conv.getIfmapEn());
        conv.setResidual(residual);
      }
      // if (!cp.coeffOnChip)
      //   conv.setCoeffList(createCoeffListInput(conv, numCoeffFifoSplits));
      // else if (cp.initCoeff) conv.initCoeff(initCoeff, initCoeffStrm, WT);

      for (int i = 0; i < cp.outputs.size(); ++i) {
        OutputType outputType = cp.outputs.get(i).type;
        getManager().logMsg("Connecting to output: %s\n", getOfmapName(i));

        if (outputType == OutputType.IFMAP) {
          // if (cp.K != 1 && cp.STRIDE == 2)
          //   throw new IllegalArgumentException("Cannot duplicate S = 2 ifmap if K != 1");
          if (cp.seq != CompSeq.FILTER_MAJOR)
            throw new IllegalArgumentException(
                "Should use FILTER_MAJOR if the ifmap will be duplicated.");
          // .connect(((ConvLayerKernel) conv).getIfmapByOfmapAddr());
        }

        io.output(getOfmapName(i), conv.getOfmapVecT(i), conv.getOfmapEn())
            .connect(conv.getOfmap(i));
      }

      // } else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
      //   DepthwiseSeparableConvLayerKernel conv =
      //       new DepthwiseSeparableConvLayerKernel(getKernel(), cp, T, WT);

      //   DFEVector<DFEVar> ifmap = io.input(IFMAP_NAME, conv.getIfmapVecT(), conv.getIfmapEn());
      //   if (!cp.coeffOnChip) {
      //     DFEVector<DFEVar> depthwiseCoeff = io.input(
      //         DEPTHWISE_COEFF_NAME, conv.getDepthwiseCoeffVecT(), conv.getDepthwiseCoeffEn());
      //     DFEVector<DFEVar> pointwiseCoeff = io.input(
      //         POINTWISE_COEFF_NAME, conv.getPointwiseCoeffVecT(), conv.getPointwiseCoeffEn());
      //     conv.setInputs(ifmap, depthwiseCoeff, pointwiseCoeff);
      //   } else {
      //     conv.setInputs(ifmap);
      //   }
      //   io.output(OFMAP_NAME, conv.getOfmapVecT(), conv.getOfmapEn()).connect(conv.getOfmap());

      //   if (cp.dbg) {
      //     debug.simPrintf("[ConvLayerWrapKernel] pointwise en = %d\n",
      //     conv.getPointwiseCoeffEn());
      //   }

    } else if (cp.type == Type.DEPTHWISE_SEPARABLE_V2) {
      throw new IllegalArgumentException("Depthwise separable v2 is not maintained.");

      // create the kernel for depthwise separable convolution V2
      // getManager().logMsg("Initializing kernel for Depthwise Separable Convolution
      // V2");

      // DepthwiseSeparableConvLayerKernelV2 conv =
      // new DepthwiseSeparableConvLayerKernelV2(getKernel(), cp, T);
      // conv.setIO(getKernel());
    } else {
      throw new IllegalArgumentException("type has no been supported");
    }
  }

  public static String getIfmapName(int index) {
    if (index == 0)
      return IFMAP_NAME;
    return IFMAP_NAME + "_" + Integer.toString(index);
  }

  public static String getOfmapName(int index) {
    if (index == 0)
      return OFMAP_NAME;
    return OFMAP_NAME + "_" + Integer.toString(index);
  }

  public List<DFEVector<DFEVar>> createCoeffListInput(
      BaseConvLayerKernel conv, int numCoeffFifoSplits) {
    List<DFEVector<DFEVar>> coeffList = new ArrayList<DFEVector<DFEVar>>();
    List<DFEVectorType<DFEVar>> coeffVecTList = conv.getCoeffVecTList();
    List<Integer> coeffVecSizeList = conv.getCoeffVecSizeList();
    List<DFEVar> coeffEnList = conv.getCoeffEnList();

    int numCoeff = coeffVecSizeList.size();

    for (int c = 0; c < numCoeff; c++) {
      int coeffVecSize = coeffVecSizeList.get(c);
      DFEVectorType<DFEVar> coeffVecT = coeffVecTList.get(c);
      DFEVar coeffEn = coeffEnList.get(c);

      // We try to reduce the size of each FIFO
      if (numCoeffFifoSplits > 1) {
        if (coeffVecSize % numCoeffFifoSplits != 0)
          throw new IllegalArgumentException(
              String.format("Coefficient vector size %d should be divisible by FIFO split size %d.",
                  coeffVecSize, numCoeffFifoSplits));

        int splitVecSize = coeffVecSize / numCoeffFifoSplits;
        getManager().logMsg(String.format("Split coefficient stream: vec size %d "
                + "FIFO splits %d split vec size %d.",
            coeffVecSize, numCoeffFifoSplits, splitVecSize));

        DFEVectorType<DFEVar> CT = new DFEVectorType<DFEVar>(T, splitVecSize);

        DFEVector<DFEVar> coeff = coeffVecT.newInstance(this);
        for (int i = 0; i < numCoeffFifoSplits; i++) {
          DFEVector<DFEVar> tmp = io.input(COEFF_NAME + "_" + i, CT, coeffEn);
          for (int j = 0; j < splitVecSize; j++)
            coeff.get(i * splitVecSize + j).connect(tmp.get(j));
        }
        coeffList.add(coeff);
      } else {
        // Just use the normal way
        coeffList.add(io.input(COEFF_NAME, coeffVecT, coeffEn));
      }
    }

    return coeffList;
  }
}
