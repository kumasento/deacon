/**
 *
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradIfmapTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradInverseTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradWeightsTransform;
import com.custom_computing_ic.maxdeep.lib.DotProductKernel;
import com.custom_computing_ic.maxdeep.utils.AdderTree;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import java.util.ArrayList;
import java.util.List;

/**
 * This kernel implements the conv2d block.
 *
 * INTERFACE: *described in ruby* < <<X(1 ,1), X(1 ,2), ..., X(1 ,K*(K+PK))>,
 * <X(2 ,1), X(2 ,2), ..., X(2 ,K*(K+PK))>, ..., <X(PC,1), X(PC,2), ...,
 * X(PC,K*(K+PK))>>_PF, <<W(1 ,1), W(1 ,2), ..., W(1 ,K*K)>, <W(2 ,1), W(2 ,2),
 * ..., W(2 ,K*K)>, ..., <W(PC,1), W(PC,2), ..., W(PC,K*K)>>_PF > ~ <<Y(1 ,1),
 * Y(1 ,2), ..., Y(1 ,PK)>, <Y(2 ,1), Y(2 ,2), ..., Y(2 ,PK)>, ..., <Y(PF,1),
 * Y(PF,2), ..., Y(PF,PK)>>
 *
 * @author Ruizhe Zhao
 *
 */
public class Conv2DKernel extends KernelComponent {
  protected final ConvLayerParameters cp;

  protected final DFEVectorType<DFEVar> ifmapVecT, coeffVecT, ofmapVecT;
  protected final DFEVector<DFEVar> ifmap, coeff, ofmap;
  protected final DFEType T, WT;

  private static final boolean OPTIMISE_WINOGRAD_TRANSFORM = true;
  private int C;

  /**
   * constructor
   *
   * @param owner
   * @param cp
   * @param T
   */
  public Conv2DKernel(KernelBase<?> owner, ConvLayerParameters cp, DFEType T, int index) {
    this(owner, cp, T, T, index);
  }

  public Conv2DKernel(
      KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT, int index) {
    this(owner, cp, T, WT, index, 0);
  }

  public Conv2DKernel(
      KernelBase<?> owner, ConvLayerParameters cp, DFEType T, DFEType WT, int index, int index2) {
    super(owner);

    this.cp = cp;
    this.C = cp.padC();

    int K = cp.K;
    int PC = cp.PC.get(index);
    int PF = cp.PF.get(index2);
    int PK = cp.PK;

    owner.getManager().logMsg(
        String.format("Building the CORE arithmetic unit for \"%s\" ...", cp.name));
    owner.getManager().logMsg("WT = %s", WT.toString());
    if (cp.useWinograd)
      owner.getManager().logMsg("WINOGRAD is applied");

    /**
     * Verify parameters.
     */
    if (cp.useWinograd && cp.PK != 1)
      throw new IllegalArgumentException("P_K should be 1 if WINOGRAD is applied, got: " + cp.PK);
    if (cp.useWinograd && cp.K != WinogradTransform.R)
      throw new IllegalArgumentException(
          String.format("Kernel size (%3d) should equal to R (%3d) in the "
                  + "WINOGRAD transform.",
              cp.K, WinogradTransform.R));

    /**
     * Setup size of vectors in interface.
     */
    int coeffVecSize = cp.getCoeffVecSize(index);
    int ifmapVecSize = (cp.useWinograd)
        ? (WinogradTransform.TILE_SIZE * WinogradTransform.TILE_SIZE * cp.PC.get(index))
        : (K * (K + PK - 1) * PC);
    int ofmapVecSize = (cp.useWinograd)
        ? WinogradTransform.M * WinogradTransform.M * cp.PF.get(index2)
        : (cp.type == Type.DEPTHWISE_SEPARABLE ? PK * PC : PK * PF);
    owner.getManager().logMsg(String.format("CORE ifmap vector size: %d", ifmapVecSize));
    owner.getManager().logMsg(String.format("CORE coefficient vector size: %d", coeffVecSize));
    owner.getManager().logMsg(String.format("CORE ofmap vector size: %d", ofmapVecSize));

    this.T = T;
    this.WT = WT;

    this.ifmapVecT = new DFEVectorType<DFEVar>(T, ifmapVecSize);
    this.coeffVecT = new DFEVectorType<DFEVar>(WT, coeffVecSize);
    this.ofmapVecT = new DFEVectorType<DFEVar>(getOfmapScalarT(), ofmapVecSize);

    this.ifmap = ifmapVecT.newInstance(owner);
    this.coeff = coeffVecT.newInstance(owner);
    this.ofmap = ofmapVecT.newInstance(owner);

    /**
     * computation
     */
    if (cp.useWinograd)
      computeWinograd(ifmap, coeff, ofmap, index, index2);
    else
      compute(ifmap, coeff, ofmap, index, index2);

    if (cp.dbg) {
      debug.simPrintf("[Conv2DKernel] ifmap = %KObj%\n", ifmap);
      debug.simPrintf("[Conv2DKernel] coeff = %KObj%\n", coeff);
      debug.simPrintf("[Conv2DKernel] ofmap = %KObj%\n", ofmap);
    }
  }

  public DFEType getOfmapScalarT() {
    return T;
  }

  private void processFilter(int pf, int PC, int PK, int ci) {
    List<DFEVar> tmpResults = new ArrayList<DFEVar>(PK * PC);
    // TODO: initialise an ArrayList in this way might be silly
    for (int i = 0; i < PC * PK; i++) tmpResults.add(null);

    for (int pc = 0; pc < PC; pc++) {
      // create a new vector instance
      List<DFEVector<DFEVar>> ifmapChunks = getIfmapChunksAt(pc, ci);

      // coefficient chunk
      DFEVector<DFEVar> coeffChunk = getCoeffChunkAt(pf, pc, ci);

      for (int pk = 0; pk < PK; pk++) {
        DFEVector<DFEVar> ifmapChunk = ifmapChunks.get(pk);

        tmpResults.set(pk * PC + pc, dotprod(ifmapChunk, coeffChunk));
      }
    }

    for (int pk = 0; pk < PK; pk++)
      ofmap.get(pf * PK + pk).connect(AdderTree.reduce(tmpResults.subList(pk * PC, (pk + 1) * PC)));
  }

  private void processPackedFilter(int pf, int PC, int ci) {
    List<DFEVar> tmpResults = new ArrayList<DFEVar>(PC);
    for (int pc = 0; pc < PC; ++pc) {
      DFEVector<DFEVar> coeffChunkA = getCoeffChunkAt(pf, pc, ci);
      DFEVector<DFEVar> coeffChunkB = getCoeffChunkAt(pf + 1, pc, ci);

      DFEVectorType<DFEVar> T = new DFEVectorType<DFEVar>(dfeInt(27), coeffChunkA.getSize());
      // (A << 18) + B
      DFEVector<DFEVar> coeffChunk = coeffChunkA.cast(T).shiftLeft(18).add(coeffChunkB.cast(T));

      // create a new vector instance
      DFEVector<DFEVar> ifmapChunk = getIfmapChunksAt(pc, ci).get(0);

      DFEVar tmp = dotprod(ifmapChunk, coeffChunk, this.T, dfeInt(27));
      if (cp.dbg) {
        debug.simPrintf("coeffChunkA = %KObj%\n", coeffChunkA);
        debug.simPrintf("coeffChunkB = %KObj%\n", coeffChunkB);
        debug.simPrintf("coeffChunk  = %KObj%\n", coeffChunk);
        debug.simPrintf("ifmapChunk  = %KObj%\n", ifmapChunk);
        debug.simPrintf("tmp         = %KObj%\n", tmp);
      }
      tmpResults.add(pc, tmp);
    }

    DFEVar acc = AdderTree.reduce(tmpResults);
    DFEType T = dfeInt(8);
    optimization.pushDSPFactor(0);
    DFEVar accB = acc.and(constant.var(255).cast(acc.getType())).cast(T);
    DFEVar accA =
        (acc.shiftRight(18))
            .cast(T)
            .add(acc.get(15).cast(dfeBool()) ? constant.var(1).cast(T) : constant.var(0).cast(T));
    optimization.popDSPFactor();
    if (cp.dbg) {
      debug.simPrintf("acc  = %KObj%\n", acc);
      debug.simPrintf("accA = %KObj% acc >> 18 = %KObj% acc.slice() = %KObj%\n", accA,
          acc.shiftRight(18), acc.slice(18, 8));
      debug.simPrintf("accB = %KObj%\n", accB);
    }
    ofmap.get(pf + 1).connect(accB);
    ofmap.get(pf).connect(accA);
  }

  /**
   * core computation, based on dot-product.
   *
   * @param ifmap
   * @param coeff
   * @param ofmap
   */
  public void compute(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff, DFEVector<DFEVar> ofmap,
      int index, int index2) {
    int K = cp.K;
    int PC = cp.PC.get(index);
    int PF = cp.PF.get(index2);
    int PK = cp.PK;

    getOwner().optimization.pushDSPFactor(cp.dspFactor);
    if (cp.type == Type.DEPTHWISE_SEPARABLE) {
      for (int pc = 0; pc < PC; pc++) {
        // create a new vector instance
        List<DFEVector<DFEVar>> ifmapChunks = getIfmapChunksAt(pc, index);

        // coefficient chunk
        DFEVector<DFEVar> coeffChunk = getCoeffChunkAt(0, pc, index);

        for (int pk = 0; pk < PK; pk++) {
          DFEVector<DFEVar> ifmapChunk = ifmapChunks.get(pk);

          ofmap.get(pc * cp.PK + pk).connect(dotprod(ifmapChunk, coeffChunk));
        }
      }

      // for (int pk = 0; pk < PK; pk++)
      //   ofmap[pf * PK + pk].connect(AdderTree.reduce(tmpResults.subList(pk * PC, (pk + 1) *
      //   PC)));
    } else {
      if (PK == 1 && cp.BW == 8 && cp.WBW == 8) {
        getOwner().getManager().logMsg("Enabled DSP optimisation for MACC.");

        for (int pf = 0; pf < PF; pf += 2) {
          if (pf + 1 == PF)
            processFilter(pf, PC, PK, index);
          else
            processPackedFilter(pf, PC, index);
        }

      } else {
        for (int pf = 0; pf < PF; pf++) processFilter(pf, PC, PK, index);
      }
    }
    getOwner().optimization.popDSPFactor();
  }

  public void computeWinograd(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff,
      DFEVector<DFEVar> ofmap, int index, int index2) {
    getOwner().getManager().logMsg("Using WINOGRAD compute function ...");
    if (cp.winogradWeightsOffline)
      getOwner().getManager().logMsg("Winograd weights are computed offline");

    // transform ifmap and coefficient
    DFEVector<DFEVar> ifmapWino = createWinogradTransformedIfmap(ifmap, index);
    DFEVector<DFEVar> coeffWino =
        (cp.winogradWeightsOffline) ? coeff : createWinogradTransformedCoeff(coeff, index, index2);

    if (cp.dbg) {
      debug.simPrintf("[Conv2DKernel] ifmap WINO = %KObj%\n", ifmapWino);
      debug.simPrintf("[Conv2DKernel] coeff WINO = %KObj%\n", coeffWino);
    }

    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    int M = WinogradTransform.M;

    // type of temporary output
    DFEVectorType<DFEVar> TOT = new DFEVectorType<DFEVar>(T, TILE_SIZE * TILE_SIZE);
    // type of output
    DFEVectorType<DFEVar> OT = new DFEVectorType<DFEVar>(T, cp.PF.get(index2) * M * M);

    getOwner().getManager().logMsg(String.format("CORE consume %d number of multipliers ...",
        cp.PC.get(index) * cp.PF.get(index2) * TILE_SIZE * TILE_SIZE));

    for (int f = 0; f < cp.PF.get(index2); f++) {
      DFEVector<DFEVar> TO = TOT.newInstance(getOwner());

      for (int c = 0; c < cp.PC.get(index); c++) {
        getOwner().getManager().logMsg(String.format(
            "CORE setting up element-wise multiply (#%03d) ...", f * cp.PC.get(index) + c));

        DFEVector<DFEVar> tmp = TOT.newInstance(getOwner());

        if (cp.dbg) {
          debug.simPrintf("[Conv2DKernel] ifmap x coeff = %KObj%\n", tmp);
        }

        for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++) {
          int ifmapIdx = c * TILE_SIZE * TILE_SIZE + j;
          int coeffIdx = (f * cp.PC.get(index) + c) * TILE_SIZE * TILE_SIZE + j;

          // getOwner().optimization.pushDSPFactor(1.0);
          DFEVar r = ifmapWino[ifmapIdx] * coeffWino[coeffIdx];
          // getOwner().optimization.popDSPFactor();

          tmp[j].connect(r);
        }

        if (c == 0)
          TO.connect(tmp);
        else
          TO += tmp;
      }

      if (cp.dbg) {
        debug.simPrintf("[Conv2DKernel] o = %KObj%\n", TO);
      }

      WinogradInverseTransform transform =
          new WinogradInverseTransform(getOwner(), T, OPTIMISE_WINOGRAD_TRANSFORM, cp.dbg);
      transform.setInputMatrix(TO);
      DFEVector<DFEVar> trans = transform.getOutput();

      for (int j = 0; j < M * M; j++) ofmap[f * M * M + j].connect(trans[j]);
    }
  }

  public DFEVector<DFEVar> createWinogradTransformedIfmap(DFEVector<DFEVar> ifmap, int index) {
    int TILE_SIZE = WinogradTransform.TILE_SIZE;
    DFEVectorType<DFEVar> RT =
        new DFEVectorType<DFEVar>(T, cp.PC.get(index) * TILE_SIZE * TILE_SIZE);

    getOwner().getManager().logMsg(String.format(
        "CORE initialised WINOGRAD transformed ifmap %d x %d x %d", cp.PC, TILE_SIZE, TILE_SIZE));
    DFEVector<DFEVar> R = RT.newInstance(getOwner());

    for (int i = 0; i < cp.PC.get(index); i++) {
      WinogradIfmapTransform winogradTransform =
          new WinogradIfmapTransform(getOwner(), T, OPTIMISE_WINOGRAD_TRANSFORM, cp.dbg);

      // the original input, split into TILE_SIZE^2 tiles
      DFEVector<DFEVar> input = winogradTransform.getInputT().newInstance(getOwner());
      getOwner().getManager().logMsg(
          String.format("CORE initialised WINOGRAD transform input matrix "
                  + "(#%04d) of length %d",
              i, input.getSize()));

      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        input[j].connect(ifmap[i * TILE_SIZE * TILE_SIZE + j]);
      winogradTransform.setInput(input);

      // transformed results
      DFEVector<DFEVar> trans = winogradTransform.getOutput();
      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        R[i * TILE_SIZE * TILE_SIZE + j].connect(trans[j]);
    }

    return R;
  }

  public DFEVector<DFEVar> createWinogradTransformedCoeff(
      DFEVector<DFEVar> coeff, int index, int index2) {
    int TILE_SIZE = WinogradTransform.TILE_SIZE;

    DFEVectorType<DFEVar> RT =
        new DFEVectorType<DFEVar>(T, cp.PF.get(index2) * cp.PC.get(index) * TILE_SIZE * TILE_SIZE);
    DFEVector<DFEVar> R = RT.newInstance(getOwner());
    getOwner().getManager().logMsg(
        String.format("CORE initialised WINOGRAD transformed coeff %s x %s x %d x %d", cp.PF, cp.PC,
            TILE_SIZE, TILE_SIZE));

    for (int i = 0; i < cp.PF.get(index2) * cp.PC.get(index); i++) {
      WinogradWeightsTransform winogradTransform =
          new WinogradWeightsTransform(getOwner(), T, OPTIMISE_WINOGRAD_TRANSFORM);

      DFEVector<DFEVar> input = winogradTransform.getInputT().newInstance(getOwner());
      getOwner().getManager().logMsg(
          String.format("CORE initialised WINOGRAD transform input matrix "
                  + "(#%04d) of length %d",
              i, input.getSize()));
      for (int j = 0; j < cp.K * cp.K; j++) input[j].connect(coeff[i * cp.K * cp.K + j]);

      winogradTransform.setInput(input);
      DFEVector<DFEVar> trans = winogradTransform.getOutput();
      for (int j = 0; j < TILE_SIZE * TILE_SIZE; j++)
        R[i * TILE_SIZE * TILE_SIZE + j].connect(trans[j]);
    }

    return R;
  }

  public DFEVar dotprod(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff) {
    return dotprod(ifmap, coeff, T, WT);
  }

  public DFEVar dotprod(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff, DFEType T, DFEType WT) {
    DotProductKernel dp = new DotProductKernel(this.getOwner(), cp.K * cp.K, T, WT, cp.dbg);
    dp.setInputs(ifmap, coeff);

    return dp.getOutput();
  }

  public DFEVectorType<DFEVar> getIfmapT() {
    return ifmapVecT;
  }

  public DFEVectorType<DFEVar> getCoeffT() {
    return coeffVecT;
  }

  public DFEVectorType<DFEVar> getOfmapT() {
    return ofmapVecT;
  }

  public void setInputs(DFEVector<DFEVar> ifmap, DFEVector<DFEVar> coeff) {
    this.ifmap.connect(ifmap);
    this.coeff.connect(coeff);
  }

  public DFEVector<DFEVar> getOfmap() {
    return ofmap;
  }

  /**
   * Get ifmap data chunk at (pc)
   *
   * TODO: Each time you call this function, a new vector will be created,
   * might need caching in the future.
   *
   * @param pc
   * @return
   */
  private List<DFEVector<DFEVar>> getIfmapChunksAt(int pc, int index) {
    int K = cp.K;
    int PC = cp.PC.get(index);
    int PK = cp.PK;

    if (pc >= PC)
      throw new IllegalArgumentException("pc should be smaller than PC");

    int ifmapPackedChunkSize = K * (K + PK - 1);

    DFEVectorType<DFEVar> ifmapPackedChunkT = new DFEVectorType<DFEVar>(T, ifmapPackedChunkSize);

    DFEVector<DFEVar> ifmapPackedChunk = ifmapPackedChunkT.newInstance(getOwner());

    // create packed chunk
    for (int i = 0; i < ifmapPackedChunkSize; i++)
      ifmapPackedChunk[i].connect(ifmap[pc * ifmapPackedChunkSize + i]);

    // unpack
    return unpackIfmapChunk(ifmapPackedChunk, cp);
  }

  private DFEVector<DFEVar> getCoeffChunkAt(int pf, int pc, int index) {
    int K = cp.K;
    int PC = cp.PC.get(index);

    int coeffChunkSize = K * K;

    DFEVectorType<DFEVar> coeffChunkT = new DFEVectorType<DFEVar>(WT, coeffChunkSize);
    DFEVector<DFEVar> coeffChunk = coeffChunkT.newInstance(getOwner());

    for (int i = 0; i < coeffChunkSize; i++)
      coeffChunk[i].connect(coeff[pf * PC * coeffChunkSize + pc * coeffChunkSize + i]);

    return coeffChunk;
  }

  /**
   * Convert vector from (K, K + PK -1) to (PK, K, K)
   *
   * @param src
   *            source vector
   * @param cp
   *
   * @author Ruizhe Zhao
   */
  private List<DFEVector<DFEVar>> unpackIfmapChunk(DFEVector<DFEVar> src, ConvLayerParameters cp) {
    int K = cp.K;
    int PK = cp.PK;

    int ifmapChunkSize = K * K;
    DFEVectorType<DFEVar> ifmapChunkT = new DFEVectorType<DFEVar>(T, ifmapChunkSize);
    List<DFEVector<DFEVar>> ifmapChunks = new ArrayList<DFEVector<DFEVar>>(PK);

    for (int p = 0; p < PK; p++) {
      ifmapChunks.add(p, ifmapChunkT.newInstance(getOwner()));

      for (int kx = 0; kx < K; kx++) {
        for (int ky = 0; ky < K; ky++) {
          int srcIdx = kx * (K + PK - 1) + (ky + p);
          int dstIdx = kx * K + ky;
          ifmapChunks.get(p)[dstIdx].connect(src[srcIdx]);
        }
      }
    }

    return ifmapChunks;
  }
}
