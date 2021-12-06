package com.custom_computing_ic.maxdeep.kernel.pool;

public class PoolingLayerParameters {

  public final int K;

  public enum Mode {
    AVG, MAX
  };

  public final Mode mode;

  public final int  S;

  public PoolingLayerParameters(int K, int S, Mode mode) {
    this.K = K;
    this.S = S;
    this.mode = mode;
  }
}
