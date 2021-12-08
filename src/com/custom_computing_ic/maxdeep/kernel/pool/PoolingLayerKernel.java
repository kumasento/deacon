package com.custom_computing_ic.maxdeep.kernel.pool;

import java.util.ArrayList;
import java.util.List;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;

public class PoolingLayerKernel extends Kernel {
  public static final String           IFMAP_NAME = "ifmap";
  public static final String           OFMAP_NAME = "ofmap";

  private final ConvLayerParameters    cp;
  private final PoolingLayerParameters pool;

  private final DFEType                T;

  private final DFEVar                 c, h, w, sh, sw, rh, rw;

  public PoolingLayerKernel(KernelParameters params, ConvLayerParameters cp) {
    super(params);

    this.cp = cp;

    this.pool = cp.pool;
    if (pool.K <= 0)
      throw new IllegalArgumentException(
          "Kernel size of pooling should be larger than 0");
    if (cp.OH % pool.K != 0)
      throw new IllegalArgumentException("OH % K should equal 0");
    if (cp.OW % pool.K != 0)
      throw new IllegalArgumentException("OW % K should equal 0");
    if (pool.K != 2 || pool.S != 2)
      throw new IllegalArgumentException("Only 2x2 pooling is supported");
    if (cp.PK != 1 && cp.PK != 2)
      throw new IllegalArgumentException("PK should equal either 1 or 2");

    this.T = dfeFix(cp.BW, 0, SignMode.TWOSCOMPLEMENT);
    DFEType countT = dfeInt(32);

    /* Setup counters */
    CounterChain chain = control.count.makeCounterChain();

    if (cp.F / cp.PF == 1)
      c = constant.var(0).cast(countT);
    else
      c = chain.addCounter(cp.F / cp.PF, 1).cast(countT);

    // height counters
    h = chain.addCounter(cp.OH, pool.S).cast(countT);
    sh = chain.addCounter(pool.S, 1).cast(countT);
    rh = h + sh;

    // width counters
    if (cp.PK == 1) {
      w = chain.addCounter(cp.OW, pool.S).cast(countT);
      sw = chain.addCounter(pool.S, 1).cast(countT);
      rw = w + sw;
    } else {
      w = chain.addCounter(cp.OW / cp.PK, 1).cast(countT);
      sw = constant.var(0).cast(countT);
      rw = w;
    }

    /* Setup interface */
    DFEVectorType<DFEVar> ifmapVecT =
        new DFEVectorType<DFEVar>(T, cp.PK * cp.PF);
    DFEVectorType<DFEVar> ofmapVecT = new DFEVectorType<DFEVar>(T, cp.PF);

    DFEVector<DFEVar> ifmap = io.input(IFMAP_NAME, ifmapVecT);
    DFEVector<DFEVar> ofmap = ofmapVecT.newInstance(getKernel());

    for (int f = 0; f < cp.PF; f++) {
      /* create chunk */
      DFEVectorType<DFEVar> ifmapChunkT = new DFEVectorType<DFEVar>(T, cp.PK);
      DFEVector<DFEVar> ifmapChunk = ifmapChunkT.newInstance(getKernel());
      for (int k = 0; k < cp.PK; k++)
        ifmapChunk[k].connect(ifmap[f * cp.PK + k]);

      List<DFEVar> tmp = new ArrayList<DFEVar>();
      if (cp.PK == 1) {
        for (int kx = 0; kx < pool.K; kx++)
          for (int ky = 0; ky < pool.K; ky++) {
            int idx = -((pool.K - kx - 1) * cp.OW + (pool.K - ky - 1));

            tmp.add(kx * pool.K + ky, stream.offset(ifmapChunk[0], idx));
          }
      } else {
        for (int kx = 0; kx < pool.K; kx++)
          for (int ky = 0; ky < pool.K; ky++) {
            int idx = -((pool.K - kx - 1) * cp.OW / 2);

            tmp.add(kx * pool.K + ky, stream.offset(ifmapChunk[ky], idx));
          }
      }

      /* max */
      DFEVar a = (tmp.get(0) > tmp.get(1)) ? tmp.get(0) : tmp.get(1);
      DFEVar b = (tmp.get(2) > tmp.get(3)) ? tmp.get(2) : tmp.get(3);

      ofmap[f].connect((a > b) ? a : b);
    }

    DFEVar ofmapEn;
    if (cp.PK == 1)
      ofmapEn = (sh.eq(1) & sw.eq(1));
    else
      ofmapEn = (sh.eq(1));

    io.output(OFMAP_NAME, ofmapVecT, ofmapEn).connect(ofmap);

    if (cp.dbg) {
      debug.simPrintf(
          "[PoolingLayer] c = %d h = %d w = %d sh = %d sw = %d en = %d\n",
          c,
          h,
          w,
          sh,
          sw,
          ofmapEn);
    }
  }

}