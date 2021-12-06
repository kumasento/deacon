package com.custom_computing_ic.maxdeep.kernel.fc;

import com.custom_computing_ic.maxdeep.lib.LayerParameters;

public class FullyConnectedLayerParameters extends LayerParameters {

  public int    H;
  public int    W;

  public int    PC;
  public int    PR;

  public String name;

  public FullyConnectedLayerParameters(String name, int BW, int H, int W) {
    this(name, BW, H, W, 1, 1);
  }

  public FullyConnectedLayerParameters(String name, int BW, int H, int W, int PC) {
    this(name, BW, H, W, PC, 1);
  }

  public FullyConnectedLayerParameters(String name, int BW, int H, int W,
      int PC, int PR) {
    this.name = name;
    this.BW = BW;
    this.H = H;
    this.W = W;
    this.PC = PC;
    this.PR = PR;
  }

  @Override
  public int getIfmapVecSize() {
    return this.PC;
  }

  public int getIfmapStreamBitWidth() {
    return getIfmapVecSize() * BW;
  }

  @Override
  public int getCoeffVecSize() {
    return this.PC * this.PR;
  }

  public int getCoeffStreamBitWidth() {
    return getCoeffVecSize() * BW;
  }

  @Override
  public int getOfmapVecSize() {
    return this.PR;
  }

  public int getOfmapStreamBitWidth() {
    return getOfmapVecSize() * BW;
  }

  @Override
  public long getOfmapStreamNumElems() {
    return H;
  }

  @Override
  public long getIfmapStreamNumElems() {
    return W;
  }

  @Override
  public long getCoeffStreamNumElems() {
    return H * W;
  }

  public int getNumCycles() {
    return H * W / (PC * PR);
  }
}
