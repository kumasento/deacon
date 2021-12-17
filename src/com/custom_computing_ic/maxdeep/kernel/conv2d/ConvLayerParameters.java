/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d;

import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.ConvLayerLineBuffer;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.custom_computing_ic.maxdeep.kernel.pool.PoolingLayerParameters;
import com.custom_computing_ic.maxdeep.lib.LayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFETypeFactory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

/**
 * Design parameters used to build a single convolution layer.
 *
 * @author Ruizhe Zhao
 *
 */
public class ConvLayerParameters extends LayerParameters {
  public final int PC, PF, PK, PH, PW; // level of parallelisation
  public final int H; // height
  public final int W; // width
  public final int OH;
  public final int OW;
  public final int C; // number of channels
  public final int F; // number of filters
  public final int K; // kernel size
  public final int PAD; // padding size
  public final int STRIDE; // stride
  public final CompSeq seq; // computation sequence
  public final String name;
  public final boolean dbg;
  public final Type type;
  public int LRNLocalSize;
  public boolean useDRAM;
  public boolean useWinograd;
  public boolean winogradWeightsOffline;
  public final boolean coeffOnChip;
  public final boolean initCoeff;

  public static final int WINOGRAD_TILE_SIZE = 4;
  public int winoH;
  public int winoW;
  public String residual = "";
  public String coeffFile = "";

  public PoolingLayerParameters pool;

  /**
   * Computation sequence
   *
   * @author Ruizhe Zhao
   *
   */
  public enum CompSeq { FILTER_MAJOR, CHANNEL_MAJOR, PIXEL_MAJOR }

  /**
   * Type of convolution computation
   *
   * @author Ruizhe Zhao
   *
   */
  public enum Type { STANDARD, POINTWISE, DEPTHWISE_SEPARABLE, DEPTHWISE_SEPARABLE_V2, BOTTLENECK }

  public ConvLayerParameters(Builder builder) {
    // inherited from the parent
    this.BW = builder.BW;
    this.WBW = builder.WBW;
    this.dtype = builder.dtype;
    this.numFracBits = builder.numFracBits;

    this.H = builder.H;
    this.W = builder.W;
    this.C = builder.C;
    this.F = builder.F;
    this.K = builder.K;
    this.OH = builder.OH;
    this.OW = builder.OW;
    this.PC = builder.PC;
    this.PF = builder.PF;
    this.PK = builder.PK;
    this.PH = builder.PH;
    this.PW = builder.PW;
    this.PAD = builder.P;
    this.STRIDE = builder.S;
    this.seq = builder.seq;
    this.name = builder.name;
    this.dbg = builder.dbg;
    this.type = builder.type;
    this.LRNLocalSize = 0;
    this.pool = builder.pool;
    this.useDRAM = builder.useDRAM;
    this.useWinograd = builder.useWinograd;
    this.winogradWeightsOffline = builder.winogradWeightsOffline;
    this.winoH = H + ConvLayerLineBuffer.WINO_LBUF_PADDING_WIDTH;
    this.winoW = W + ConvLayerLineBuffer.WINO_LBUF_PADDING_WIDTH;
    this.coeffOnChip = builder.coeffOnChip;
    this.residual = builder.residual;
    this.initCoeff = builder.initCoeff;
    this.coeffFile = builder.coeffFile;
  }

  public ConvLayerParameters createDepthwiseParameters() {
    /**
     * TODO: One thing not clear - PC is actually not sensible here However, we
     * still need it to specify correct number of line buffers.
     */
    return new ConvLayerParameters.Builder(H, W, C, C, K)
        .dtype(dtype)
        .BW(BW)
        .WBW(WBW)
        .numFracBits(numFracBits)
        .PC(PC)
        .PK(PK)
        .coeffOnChip(coeffOnChip)
        .useDRAM(useDRAM)
        .seq(seq)
        .dbg(dbg)
        .useWinograd(useWinograd)
        .type(Type.DEPTHWISE_SEPARABLE)
        .name(name + "_dw")
        .build();
  }

  public ConvLayerParameters createPointwiseParameters() {
    return new ConvLayerParameters.Builder(OH, OW, C, F, 1)
        .dtype(dtype)
        .BW(BW)
        .WBW(WBW)
        .numFracBits(numFracBits)
        .PC(PC)
        .PF(PF)
        .PK(PK)
        .coeffOnChip(coeffOnChip)
        .useDRAM(useDRAM)
        .name(name + "_pw")
        .type(Type.STANDARD)
        .seq(seq)
        .dbg(dbg)
        .build();
  }

  /**
   * Create DFEType from ConvLayerParameters.
   *
   * @return Created DFEType instance.
   */
  public DFEType getDFEType() {
    return getDFEType(this.BW);
  }

  public DFEType getDFEType(int BW) {
    if (dtype.equals("float")) {
      if (BW != 32)
        throw new IllegalArgumentException(
            String.format("If dtype is float, BW should be 32. Got: %d", BW));

      return DFETypeFactory.dfeFloat(8, 24);
    } else if (dtype.equals("fixed")) {
      return DFETypeFactory.dfeFix(BW - numFracBits, numFracBits, SignMode.TWOSCOMPLEMENT);
    } else if (dtype.equals("int")) {
      return DFETypeFactory.dfeInt(BW);
    }

    throw new IllegalArgumentException(
        "BW " + BW + " and dtype " + dtype + " don't give a valid type definition.");
  }

  public int getPaddedHeight() {
    return H + 2 * PAD;
  }

  public int getPaddedWidth() {
    return W + 2 * PAD;
  }

  public long getNumCycles() {
    if (useWinograd) {
      int PH = WinogradTransform.M;
      int PW = WinogradTransform.M;

      return ((long) winoH * winoW * C * F) / (PC * PF * PH * PW);
    }

    long H = getPaddedHeight();
    long W = getPaddedWidth();

    if (type == Type.STANDARD)
      return ((long) H * W * C * F) / (PC * PF * PK);
    if (type == Type.DEPTHWISE_SEPARABLE)
      return ((long) H * W * C * F) / (PC * PF * PK);
    if (type == Type.DEPTHWISE_SEPARABLE_V2)
      return ((long) H * W * C) / (PC * PK) + ((long) H * W * C * F) / (PC * PF * PK);
    if (type == Type.POINTWISE)
      return ((long) H * W * C * F) / (PC * PF * PH * PW);
    throw new IllegalArgumentException(
        "getNumCycles has not implemented for the current type: " + type.name());
  }

  public int getPoolNumCycles() {
    return (OH * OW * F) / (PK * PF);
  }

  @Override
  public long getIfmapStreamNumElems() {
    return useWinograd ? C * winoH * winoW : C * H * W;
  }

  @Override
  public int getIfmapVecSize() {
    if (type == Type.POINTWISE)
      return PC * PH * PW;

    // standard
    return useWinograd ? PC * ConvLayerLineBuffer.WINO_LBUF_NUM_PIPES : PC * PK;
  }

  @Override
  public long getCoeffStreamNumElems() {
    if (useWinograd && winogradWeightsOffline)
      return C * F * WinogradTransform.TILE_SIZE * WinogradTransform.TILE_SIZE;
    if (type == Type.STANDARD)
      return C * F * K * K;
    if (type == Type.POINTWISE)
      return C * F;

    return C * (F / PF + 1) * Math.max(K * K, PF);
  }

  @Override
  public int getCoeffVecSize() {
    // The weights offline mode of Winograd will increase the vector size of
    // coefficients.
    if (useWinograd && winogradWeightsOffline)
      return PC * PF * WinogradTransform.TILE_SIZE * WinogradTransform.TILE_SIZE;
    if (type == Type.POINTWISE)
      return PC * PF;
    if (type == Type.DEPTHWISE_SEPARABLE_V2)
      return PC * Math.max(K * K, PF);
    if (type == Type.STANDARD || type == Type.DEPTHWISE_SEPARABLE)
      return (PC * PF * K * K);

    throw new IllegalArgumentException(
        "getNumCycles has not implemented for the current type: " + type.name());
  }

  public int getCoeffStreamLMemBitWidth() {
    return getCoeffLMemVecSize() * getCPUTypes().sizeInBytes() * 8;
  }

  public int getCoeffLMemVecSize() {
    return getCoeffVecSize();
  }

  public int getCoeffNumVec() {
    int total = 0;
    if (type == Type.DEPTHWISE_SEPARABLE)
      total = C * K * K;
    else if (type == Type.STANDARD || type == Type.POINTWISE)
      total = C * F * K * K;

    return total / getCoeffVecSize();
  }

  public long getDepthwiseCoeffStreamNumElems() {
    return C * K * K;
  }

  public int getDepthwiseCoeffVecSize() {
    return PC * K * K;
  }

  public int getDepthwiseCoeffLMemVecSize() {
    return PC;
  }

  public long getPointwiseCoeffStreamNumElems() {
    return C * F;
  }

  public int getPointwiseCoeffVecSize() {
    return PC * PF;
  }

  @Override
  public long getOfmapStreamNumElems() {
    return F * OH * OW;
  }

  @Override
  public int getOfmapVecSize() {
    return useWinograd ? WinogradTransform.M * WinogradTransform.M * PF : PK * PF;
  }

  public long getIfmapStreamSize() {
    long sizeInBytes = getCPUTypes().sizeInBytes();
    return sizeInBytes * getIfmapStreamNumElems();
  }

  public long getCoeffStreamSize() {
    long sizeInBytes = getCPUTypes().sizeInBytes();
    return sizeInBytes * getCoeffStreamNumElems();
  }

  public long getDepthwiseCoeffStreamSize() {
    long sizeInBytes = getCPUTypes().sizeInBytes();
    return sizeInBytes * getDepthwiseCoeffStreamNumElems();
  }

  public long getPointwiseCoeffStreamSize() {
    long sizeInBytes = getCPUTypes().sizeInBytes();
    return sizeInBytes * getPointwiseCoeffStreamNumElems();
  }

  public long getOfmapStreamSize() {
    long sizeInBytes = getCPUTypes().sizeInBytes();
    return sizeInBytes * getOfmapStreamNumElems();
  }

  public int getCoeffStreamBitWidth() {
    return getCoeffVecSize() * getCPUTypes().sizeInBytes() * 8;
  }

  public int getCoeffStreamChunkSize() {
    return getCoeffVecSize() / (PC * PF);
  }

  public DFEVectorType<DFEVar> getCoeffVecT(DFEType baseT) {
    return new DFEVectorType<DFEVar>(baseT, getCoeffVecSize());
  }

  public int getDepthwiseCoeffStreamBitWidth() {
    return getDepthwiseCoeffVecSize() * getCPUTypes().sizeInBytes() * 8;
  }

  public int getDepthwiseCoeffStreamLMemBitWidth() {
    return getDepthwiseCoeffLMemVecSize() * getCPUTypes().sizeInBytes() * 8;
  }

  public int getPointwiseCoeffStreamBitWidth() {
    return getPointwiseCoeffVecSize() * getCPUTypes().sizeInBytes() * 8;
  }

  public int getIfmapStreamBitWidth() {
    return getIfmapVecSize() * getCPUTypes().sizeInBytes() * 8;
  }

  public int getOfmapStreamBitWidth() {
    return getOfmapVecSize() * getCPUTypes().sizeInBytes() * 8;
  }

  /**
   * Builder class for ConvLayerParameters
   *
   * @author Ruizhe Zhao
   *
   */
  public static class Builder {
    private int H;
    private int W;
    private final int C;
    private final int F;
    private final int K;
    private final int OH;
    private final int OW;
    private String residual = "";
    private int BW; /* bit width */
    private int WBW;
    private int PC;
    private int PF;
    private int PK;
    private int PH;
    private int PW;
    private int S; // stride
    private int P; // pad
    private CompSeq seq;
    private String name;
    private String dtype;
    private int numFracBits;
    private Type type;
    private PoolingLayerParameters pool;
    private boolean useDRAM;
    private boolean dbg;
    private boolean useWinograd;
    private boolean winogradWeightsOffline;
    private boolean coeffOnChip;
    private boolean initCoeff;
    private String coeffFile = "";

    public Builder(int OH, int OW, int C, int F, int K) {
      if (OH <= 0)
        throw new IllegalArgumentException("OH should be larger than 0");
      if (OW <= 0)
        throw new IllegalArgumentException("OW should be larger than 0");
      if (C <= 0)
        throw new IllegalArgumentException("C should be larger than 0");
      if (F <= 0)
        throw new IllegalArgumentException("F should be larger than 0");
      if (K <= 0)
        throw new IllegalArgumentException("K should be larger than 0");

      this.BW = 8;
      this.dtype = "fixed";
      this.numFracBits = 8;

      // TODO: make S and P variable
      this.S = 1;
      this.P = 0;
      this.OH = OH;
      this.OW = OW;
      this.C = C;
      this.F = F;
      this.K = K;
      this.H = this.OH - 2 * this.P - 1 + this.K;
      this.W = this.OW - 2 * this.P - 1 + this.K;
      this.PF = 1;
      this.PC = 1;
      this.PK = 1;
      this.PH = 1;
      this.PW = 1;
      this.seq = CompSeq.CHANNEL_MAJOR;
      this.type = Type.STANDARD;
      this.useDRAM = true;
      this.useWinograd = false;
      this.winogradWeightsOffline = false;
      this.dbg = false;
      this.coeffOnChip = false;
      this.initCoeff = false;
    }

    public Builder coeffFile(String coeffFile) {
      if (!coeffOnChip)
        throw new IllegalArgumentException(
            "coeffOnChip should be set to true when coeffFile is provided.");

      this.coeffFile = coeffFile;
      return this;
    }

    private int getInputDim(int outputDim, int kernelDim, int pad, int stride) {
      if (type == Type.POINTWISE)
        return outputDim * stride;
      return (outputDim - 1) * stride + kernelDim - 2 * pad;
    }

    public Builder pad(int pad) {
      if (type == Type.POINTWISE && pad != 0)
        throw new IllegalArgumentException("pad should be 0 for pointwise");
      this.P = pad;
      this.H = getInputDim(this.OH, this.K, this.P, this.S);
      this.W = getInputDim(this.OW, this.K, this.P, this.S);
      return this;
    }

    public Builder stride(int stride) {
      this.S = stride;
      this.H = getInputDim(this.OH, this.K, this.P, this.S);
      this.W = getInputDim(this.OW, this.K, this.P, this.S);
      return this;
    }

    public Builder BW(int BW) {
      if (BW <= 0)
        throw new IllegalArgumentException("BW should be larger than 0");
      this.BW = BW;
      return this;
    }

    public Builder WBW(int WBW) {
      if (WBW <= 0)
        throw new IllegalArgumentException("WBW should be larger than 0");
      this.WBW = WBW;

      return this;
    }

    public Builder residual(String name) {
      this.residual = name;
      return this;
    }

    public Builder dtype(String dtype) {
      if (dtype.equals("float") || dtype.equals("fixed") || dtype.equals("int")) {
        this.dtype = dtype;
        return this;
      }

      throw new IllegalArgumentException(
          String.format("dtype cannot be recognised, should be one of float, "
                  + "fixed, int. Got \"%s\"",
              dtype));
    }

    public Builder numFracBits(int numFracBits) {
      if (numFracBits < 0 || numFracBits > BW)
        throw new IllegalArgumentException(
            "Number of fraction bits should >= 0 and <= bit width. Got " + numFracBits
            + " while BW is " + BW);

      this.numFracBits = numFracBits;
      return this;
    }

    public Builder PF(int PF) {
      if (PF <= 0)
        throw new IllegalArgumentException("PF should be larger than 0");
      if (F % PF != 0)
        throw new IllegalArgumentException("F % PF should equal 0");
      this.PF = PF;
      return this;
    }

    public Builder PK(int PK) {
      if (PK <= 0)
        throw new IllegalArgumentException("PK should be larger than 0");
      // TODO: we need to verify whether it is a good condition
      if (W % PK != 0)
        throw new IllegalArgumentException("W % PK should equal 0");
      if ((W - K + 1) % PK != 0)
        throw new IllegalArgumentException("OW % PK should equal 0");
      if ((K + PK - 1) % PK != 0)
        throw new IllegalArgumentException("(K + PK - 1) % PK should equal 0");
      this.PK = PK;
      return this;
    }

    public Builder PC(int PC) {
      if (PC <= 0)
        throw new IllegalArgumentException("PC should be larger than 0");
      if (C % PC != 0)
        throw new IllegalArgumentException("C % PC should equal 0");
      this.PC = PC;
      return this;
    }

    public Builder PH(int PH) {
      if (PH <= 0)
        throw new IllegalArgumentException("PH should be larger than 0, got " + PH);
      if (H % PH != 0)
        throw new IllegalArgumentException(String.format("H (%d) % PH (%d) != 0", H, PH));
      this.PH = PH;

      return this;
    }

    public Builder PW(int PW) {
      if (PW <= 0)
        throw new IllegalArgumentException("PW should be larger than 0, got " + PW);
      if (W % PW != 0)
        throw new IllegalArgumentException(String.format("W (%d) % PW (%d) != 0", W, PW));
      if (PW > 1 && P != 0)
        throw new IllegalArgumentException("Padding should be 0 if PW is larger than 1");

      this.PW = PW;

      return this;
    }

    public Builder name(String name) {
      this.name = name;

      return this;
    }

    public Builder seq(CompSeq seq) {
      this.seq = seq;

      return this;
    }

    public Builder type(Type type) {
      if (type == Type.POINTWISE && P != 0)
        throw new IllegalArgumentException("pad should be 0 for pointwise");
      if (type == Type.POINTWISE && K != 1)
        throw new IllegalArgumentException("kernel should be 1 for pointwise");
      this.type = type;

      return this;
    }

    public Builder pool(PoolingLayerParameters pool) {
      this.pool = pool;

      return this;
    }

    public Builder useDRAM(boolean useDRAM) {
      this.useDRAM = useDRAM;

      return this;
    }

    public Builder useWinograd(boolean useWinograd) {
      this.useWinograd = useWinograd;
      return this;
    }

    public Builder winogradWeightsOffline(boolean offline) {
      this.winogradWeightsOffline = offline;
      return this;
    }

    public Builder dbg(boolean dbg) {
      this.dbg = dbg;
      return this;
    }

    public Builder coeffOnChip(boolean coeffOnChip) {
      this.coeffOnChip = coeffOnChip;
      return this;
    }

    public Builder initCoeff(boolean initCoeff) {
      this.initCoeff = initCoeff;
      return this;
    }

    public ConvLayerParameters build() {
      if (WBW < 8 && !coeffOnChip)
        throw new IllegalArgumentException("coeff should be on chip if WBW < 8");

      return new ConvLayerParameters(this);
    }
  }
}
