/**
 *
 */
package com.custom_computing_ic.maxdeep.utils;

import java.util.ArrayList;
import java.util.List;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.MathUtils;

/**
 * @author Ruizhe Zhao
 *
 */
public class ArbitraryLengthPortROM extends KernelComponent {

  public static final int MAX_LENGTH_IN_BITS = 64;

  private final List<Memory<DFEVector<DFEVar>>> romList;
  private final List<DFEVectorType<DFEVar>> romVecTList;

  private final int numElems;
  private final int depth;
  private final DFEType scalarT;

  /**
   * @param owner
   */
  public ArbitraryLengthPortROM(KernelBase<?> owner, int numElems, int depth, DFEType scalarT) {
    super(owner);

    if (depth <= 1)
      throw new IllegalArgumentException("Depth should be larger than 1");
    if (MAX_LENGTH_IN_BITS % scalarT.getTotalBits() != 0)
      throw new IllegalArgumentException(
          "Number of bits of scalarT is not divisable by MAX_LENGTH_IN_BITS");

    this.numElems = numElems;
    this.depth = depth;
    this.scalarT = scalarT;

    int lengthInBits = numElems * scalarT.getTotalBits();
    int numROM = (int) Math.ceil((double)lengthInBits / MAX_LENGTH_IN_BITS);
    getOwner().getManager().logMsg("numROM: %d\n", numROM);
    romList = new ArrayList<Memory<DFEVector<DFEVar>>>();
    romVecTList = new ArrayList<DFEVectorType<DFEVar>>();

    for (int i = 0; i < numROM; i ++) {
      int currentLengthInBits = (i != numROM - 1)
	      ? MAX_LENGTH_IN_BITS
	      : (lengthInBits - i * MAX_LENGTH_IN_BITS);
      int currentNumElems = currentLengthInBits / scalarT.getTotalBits();

      DFEVectorType<DFEVar> vecT =
        new DFEVectorType<DFEVar>(scalarT, currentNumElems);
      romVecTList.add(vecT);
      Memory<DFEVector<DFEVar>> mem =
        new Memory<DFEVector<DFEVar>>(getOwner(), depth, vecT, null);
      romList.add(mem);
    }
  }

  public DFEType getAddrT() {
    int addrTotalBits = MathUtils.bitsToAddress(depth);
    return dfeUInt(addrTotalBits);
  }

  public DFEVector<DFEVar> read(DFEVar addr) {
    DFEVectorType<DFEVar> readVecT =
      new DFEVectorType<DFEVar>(scalarT, numElems);
    DFEVector<DFEVar> readVec = readVecT.newInstance(getOwner());

    int readVecIdx = 0;
    for (int i = 0; i < romList.size(); i ++) {
      DFEVector<DFEVar> romVec = romList[i].read(addr);
      for (int j = 0; j < romVec.getSize(); j ++)
        readVec[readVecIdx + j] <== romVec[j];
      readVecIdx += romVec.getSize();
    }

    return readVec;
  }

  public void mapToCPU(String prefix) {
    for (int i = 0; i < romList.size(); i ++)
      romList[i].mapToCPU(String.format("%s_%d", prefix, i));
  }

  public int getROMPortWitdh(int index) {
    return romVecTList.get(index).getTotalBits();
  }

  public int getNumROMs() {
    return romList.size();
  }
}
