/**
 * 
 */
package com.custom_computing_ic.maxdeep.kernel.conv2d.lib;

import com.custom_computing_ic.maxdeep.kernel.conv2d.lib.Conv2DFactorizedModuleParameter.ShapeMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Count;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Count.Counter;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.Mem;
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
public class Conv2DFactorizedModuleCache extends KernelComponent {

  private final DFEType                         scalarT;
  private final DFEVectorType<DFEVar>           vectorT;
  private final Conv2DFactorizedModuleParameter params;
  private final DFEVector<DFEVar> dataIn;
  private final DFEVector<DFEVar> dataOut;
  private final DFEVar writeEnable;
  private final DFEVar addr;
  private final int memSize;
  

  /**
   * @param owner
   */
  public Conv2DFactorizedModuleCache(KernelBase<?> owner,
      Conv2DFactorizedModuleParameter params, DFEType scalarT) {
    super(owner);

    this.params = params;
    if (params.getShapeMode() != ShapeMode.STATIC)
      throw new RuntimeException("shape modes other than STATIC are not supported.");
    
    this.memSize = params.getCacheTotalSize() / params.getIfmapNumParaChnl();
    this.scalarT = scalarT;
    this.vectorT = new DFEVectorType<DFEVar>(scalarT,
        params.getIfmapNumParaChnl());
    
    Memory<DFEVector<DFEVar>> mem = new Memory<DFEVector<DFEVar>>(getOwner(),
        memSize, vectorT, null);
   
    addr = getAddr();
    dataIn = vectorT.newInstance(owner);
    dataOut = vectorT.newInstance(owner);
    writeEnable = dfeBool().newInstance(owner);

    DFEVector<DFEVar> port = mem.port(
        addr,
        dataIn,
        writeEnable,
        Mem.RamWriteMode.WRITE_FIRST);
    
    dataOut <== port;
  }

  public Conv2DFactorizedModuleCache(KernelBase<?> owner,
      Conv2DProcessEngineParameters params, DFEType scalarT) {
    this(
        owner,
        new Conv2DFactorizedModuleParameter
          .StaticBuilder(params.getMaxHeight(), params.getMaxWidth(), params.getMaxNumChnl(), params.getMaxNumFltr())
          .knlHeight(params.getMaxKnlHeight())
          .knlWidth(params.getMaxKnlWidth())
          .ifmapNumParaChnl(params.getNumParaChnl())
          .ofmapNumParaChnl(params.getNumParaFltr()).build(),
        scalarT);
  }
  
  public DFEVectorType<DFEVar> getVectorT() {
    return vectorT;
  }
  
  private DFEVar getAddr() {
    // DFEType addrT = dfeUInt(MathUtils.bitsToAddress(memSize));
    Count.Params counterParams = getOwner().control.count.makeParams(MathUtils.bitsToAddress(memSize)).withMax(memSize);
    Counter counter = getOwner().control.count.makeCounter(counterParams);
    return counter.getCount();
  }
  
  public void setInput(DFEVector<DFEVar> dataIn) { this.dataIn <== dataIn; }
  
  public void setWriteEnable(DFEVar writeEnable) { this.writeEnable <== writeEnable; }
  
  public DFEVector<DFEVar> getOutput() { return dataOut; } 
}
