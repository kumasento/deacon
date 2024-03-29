package com.custom_computing_ic.maxdeep.manager;

import com.maxeler.maxcompiler.v2.build.EngineParameters;

public class ConvLayerEngineParameters extends EngineParameters {
  private static final String BIT_WIDTH_NAME = "bitWidth";
  private static final int BIT_WIDTH = 32;
  private static final String WBW_NAME = "WBW"; // weight bit-width
  private static final int WBW = 32; // weight bit-width

  private static final String PF_NAME = "PF";
  private static final int PF = 1;
  private static final String PC_NAME = "PC";
  private static final int PC = 1;
  private static final String PK_NAME = "PK";
  private static final int PK = 1;

  private static final String H_NAME = "H";
  private static final int H = 1;
  private static final String W_NAME = "W";
  private static final int W = 1;
  private static final String C_NAME = "C";
  protected static final int C = 1;
  private static final String F_NAME = "F";
  private static final int F = 1;
  private static final String K_NAME = "K";
  protected static final int K = 1;
  private static final String PAD_NAME = "PAD";
  protected static final int PAD = 0;
  private static final String STRIDE_NAME = "S";
  protected static final int STRIDE = 1;

  private static final String SEQ_NAME = "SEQ";
  private static final int SEQ = 0;
  private static final String FREQ_NAME = "FREQ";
  private static final int FREQ = 100;

  private static final String USE_DRAM_NAME = "USE_DRAM";
  private static final boolean USE_DRAM = false;
  /* decide whether to use BNN optimisation when BW = 1 */
  private static final String USE_BNN_NAME = "USE_BNN";
  private static final boolean USE_BNN = false;
  private static final String USE_WINOGRAD_NAME = "USE_WINOGRAD";
  private static boolean USE_WINOGRAD = false;
  private static final String WINOGRAD_WEIGHTS_OFFLINE_NAME = "WINOGRAD_WEIGHTS_OFFLINE";
  protected static boolean WINOGRAD_WEIGHTS_OFFLINE = false;

  private static final String NUM_COEFF_FIFO_SPLITS_NAME = "NUM_COEFF_FIFO_SPLITS";
  private static final int NUM_COEFF_FIFO_SPLITS = 1;

  private static final String DTYPE_NAME = "DTYPE";
  private static String DTYPE = "fixed";
  private static final String NUM_FRAC_BITS_NAME = "NUM_FRAC_BITS";
  private static int NUM_FRAC_BITS = 8;

  private static final String DEBUG_NAME = "DEBUG";
  private static final boolean DEBUG = false;

  private static final String COEFF_ON_CHIP_NAME = "COEFF_ON_CHIP";
  private static final boolean COEFF_ON_CHIP = false;
  private static final String INIT_COEFF_NAME = "INIT_COEFF";
  private static final boolean INIT_COEFF = false;

  // File that stores coefficient values.
  private static final String COEFF_FILE_NAME = "COEFF_FILE";
  private static final String COEFF_FILE = "";

  public ConvLayerEngineParameters(String[] args) {
    super(args);
  }

  @Override
  protected void declarations() {
    declareParam(BIT_WIDTH_NAME, DataType.INT, BIT_WIDTH);
    declareParam(WBW_NAME, DataType.INT, WBW);
    declareParam(DTYPE_NAME, DataType.STRING, DTYPE);
    declareParam(NUM_FRAC_BITS_NAME, DataType.INT, NUM_FRAC_BITS);

    declareParam(PF_NAME, DataType.INT, PF);
    declareParam(PC_NAME, DataType.INT, PC);
    declareParam(PK_NAME, DataType.INT, PK);

    declareParam(H_NAME, DataType.INT, H);
    declareParam(W_NAME, DataType.INT, W);
    declareParam(C_NAME, DataType.INT, C);
    declareParam(F_NAME, DataType.INT, F);
    declareParam(K_NAME, DataType.INT, K);
    declareParam(PAD_NAME, DataType.INT, PAD);
    declareParam(STRIDE_NAME, DataType.INT, STRIDE);

    declareParam(SEQ_NAME, DataType.INT, SEQ);
    declareParam(FREQ_NAME, DataType.INT, FREQ);

    declareParam(USE_DRAM_NAME, DataType.BOOL, USE_DRAM);
    declareParam(USE_BNN_NAME, DataType.BOOL, USE_BNN);
    declareParam(USE_WINOGRAD_NAME, DataType.BOOL, USE_WINOGRAD);
    declareParam(WINOGRAD_WEIGHTS_OFFLINE_NAME, DataType.BOOL, WINOGRAD_WEIGHTS_OFFLINE);
    declareParam(NUM_COEFF_FIFO_SPLITS_NAME, DataType.INT, NUM_COEFF_FIFO_SPLITS);

    declareParam(DEBUG_NAME, DataType.BOOL, DEBUG);
    declareParam(COEFF_ON_CHIP_NAME, DataType.BOOL, COEFF_ON_CHIP);
    declareParam(INIT_COEFF_NAME, DataType.BOOL, INIT_COEFF);
    declareParam(COEFF_FILE_NAME, DataType.STRING, COEFF_FILE);
  }

  public String getCoeffFile() {
    return getParam(COEFF_FILE_NAME);
  }

  public int getBitWidth() {
    return getParam(BIT_WIDTH_NAME);
  }

  public int getWBW() {
    return getParam(WBW_NAME);
  }

  public int getNumFracBits() {
    return getParam(NUM_FRAC_BITS_NAME);
  }

  public String getDType() {
    return getParam(DTYPE_NAME);
  }

  public int getPF() {
    return getParam(PF_NAME);
  }

  public int getPC() {
    return getParam(PC_NAME);
  }

  public int getPK() {
    return getParam(PK_NAME);
  }

  public int getH() {
    return getParam(H_NAME);
  }

  public int getW() {
    return getParam(W_NAME);
  }

  public int getC() {
    return getParam(C_NAME);
  }

  public int getF() {
    return getParam(F_NAME);
  }

  public int getK() {
    return getParam(K_NAME);
  }

  public int getPad() {
    return getParam(PAD_NAME);
  }

  public int getStride() {
    return getParam(STRIDE_NAME);
  }

  public int getSeq() {
    return getParam(SEQ_NAME);
  }

  public int getFreq() {
    return getParam(FREQ_NAME);
  }

  public boolean getUseDRAM() {
    return getParam(USE_DRAM_NAME);
  }

  public boolean getUseBNN() {
    return getParam(USE_BNN_NAME);
  }

  public boolean getUseWinograd() {
    return getParam(USE_WINOGRAD_NAME);
  }

  public boolean getWinogradWeightsOffline() {
    return getParam(WINOGRAD_WEIGHTS_OFFLINE_NAME);
  }

  public int getNumCoeffFifoSplits() {
    return getParam(NUM_COEFF_FIFO_SPLITS_NAME);
  }

  public boolean getDebug() {
    return getParam(DEBUG_NAME);
  }

  public boolean getCoeffOnChip() {
    return getParam(COEFF_ON_CHIP_NAME);
  }

  public boolean getInitCoeff() {
    return getParam(INIT_COEFF_NAME);
  }

  @Override
  protected void validate() {
    if (getBitWidth() <= 0)
      throw new IllegalArgumentException("bitWidth should be larger than 0.");
    if (getPF() <= 0)
      throw new IllegalArgumentException("PF should be larger than 0.");
    if (getPC() <= 0)
      throw new IllegalArgumentException("PC should be larger than 0.");
    if (getPK() <= 0)
      throw new IllegalArgumentException("PK should be larger than 0.");
    if (getH() <= 0)
      throw new IllegalArgumentException("H should be larger than 0.");
    if (getW() <= 0)
      throw new IllegalArgumentException("W should be larger than 0.");
    if (getC() <= 0)
      throw new IllegalArgumentException("C should be larger than 0.");
    if (getF() <= 0)
      throw new IllegalArgumentException("F should be larger than 0.");
    if (getK() <= 0)
      throw new IllegalArgumentException("K should be larger than 0.");
    if (getSeq() < 0 && getSeq() <= 2)
      throw new IllegalArgumentException(
          "SEQ should be larger than or equal to 0 and smaller than 3.");
    if (getBitWidth() != 1 && getUseBNN())
      throw new IllegalArgumentException("BNN OPT should not be used if BW != 1");
  }

  @Override
  public String getBuildName() {
    return String.format("%s_%s_%s_b%d_H%d_W%d_C%d_F%d_K%d_f%d_c%d_k%d_SEQ%d_%s-COEFF_%s",
        getMaxFileName(), getDFEModel(), getTarget(), getBitWidth(), getH(), getW(), getC(), getF(),
        getK(), getPF(), getPC(), getPK(), getSeq(), (getUseDRAM() ? "DRAM" : "PCIe"),
        (getCoeffOnChip() ? "FMEM" : "LMEM"));
  }
}
