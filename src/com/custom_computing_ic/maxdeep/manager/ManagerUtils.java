// Copied from dfe_snippet
package com.custom_computing_ic.maxdeep.manager;

import com.custom_computing_ic.maxdeep.manager.ManagerInterface;

import com.maxeler.maxcompiler.v2.managers.custom.stdlib.*;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.build.EngineParameters;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface.*;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.DFEModel;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;

// import static
// com.maxeler.maxcompiler.v2.managers.custom.ManagerInterface.LMemFrequency.*;

public
class ManagerUtils {

  //  public
  //   static void setDRAMMaxDeviceFrequency(ManagerInterface manager,
  //                                         EngineParameters ep) {
  //     if (ep.getDFEModel() == DFEModel.MAIA) {
  //       setDRAMFreq(manager, ep, 800);
  //     } else {
  //       setDRAMFreq(manager, ep, 400);
  //     }
  //   }

  //  public
  //   static void setDRAMFreq(ManagerInterface manager, EngineParameters ep,
  //                           int freq) {
  //     MemoryControllerConfig memCfg = new MemoryControllerConfig();
  //     ManagerInterface.LMemFrequency devFreq;
  //     if (ep.getDFEModel() == DFEModel.MAIA) {
  //       memCfg.setEnableParityMode(true, true, 72, false);
  //       if (freq > 400) {
  //         // higher frequencies require parity mode, quarter rate mode and
  //         // additional pipelining
  //         memCfg.setMAX4qMode(true);
  //         memCfg.setDataReadFIFOExtraPipelineRegInFabric(true);
  //       }
  //       switch (freq) {
  //         case 400:
  //           devFreq = MAX4MAIA_400;
  //           break;
  //         case 533:
  //           devFreq = MAX4MAIA_533;
  //           break;
  //         case 666:
  //           devFreq = MAX4MAIA_666;
  //           break;
  //         case 733:
  //           devFreq = MAX4MAIA_733;
  //           break;
  //         case 800:
  //           devFreq = MAX4MAIA_800;
  //           break;
  //         default:
  //           throw new RuntimeException("Unsupported memory frequency " + freq
  //           +
  //                                      " for device mode " +
  //                                      ep.getDFEModel());
  //       }
  //     } else {
  //       switch (freq) {
  //         case 300:
  //           devFreq = MAX3_300;
  //           break;
  //         case 333:
  //           devFreq = MAX3_333;
  //           break;
  //         case 350:
  //           devFreq = MAX3_350;
  //           break;
  //         case 400:
  //           devFreq = MAX3_400;
  //           break;
  //         default:
  //           throw new RuntimeException("Unsupported memory frequency " + freq
  //           +
  //                                      " for device mode " +
  //                                      ep.getDFEModel());
  //       }
  //     }
  //     manager.config.setOnCardMemoryFrequency(devFreq);
  //     manager.config.setMemoryControllerConfig(memCfg);
  //   }

  // A generic DRAM write interface
 public
  static EngineInterface interfaceWrite(String name, String fromCpuStream,
                                        String cpu2lmemStream) {
    EngineInterface ei = new EngineInterface(name);
    CPUTypes TYPE = CPUTypes.INT;
    InterfaceParam size = ei.addParam("size_bytes", TYPE);
    InterfaceParam start = ei.addParam("start_bytes", TYPE);
    ei.setStream(fromCpuStream, CPUTypes.UINT8, size);
    ei.setLMemLinear(cpu2lmemStream, start, size);
    ei.ignoreAll(Direction.IN_OUT);
    return ei;
  }

 public
  static EngineInterface dramWrite(ManagerInterface m, LMemInterface iface) {
    iface
        .addStreamToLMem("cpu2lmem",
                         LMemCommandGroup.MemoryAccessPattern.LINEAR_1D)
        .connect(m.addStreamFromCPU("fromcpu"));
    EngineInterface ei = new EngineInterface("dramWrite");
    CPUTypes TYPE = CPUTypes.INT;
    InterfaceParam size = ei.addParam("size_bytes", TYPE);
    InterfaceParam start = ei.addParam("start_bytes", TYPE);
    ei.setStream("fromcpu", CPUTypes.UINT8, size);
    ei.setLMemLinear("cpu2lmem", start, size);
    ei.ignoreAll(Direction.IN_OUT);
    return ei;
  }

  // A generic DRAM read interface
 public
  static EngineInterface interfaceRead(String name, String toCpuStream,
                                       String lmem2cpuStream) {
    EngineInterface ei = new EngineInterface(name);
    CPUTypes TYPE = CPUTypes.INT;
    InterfaceParam size = ei.addParam("size_bytes", TYPE);
    InterfaceParam start = ei.addParam("start_bytes", TYPE);
    ei.setStream(toCpuStream, CPUTypes.UINT8, size);
    ei.setLMemLinear(lmem2cpuStream, start, size);
    ei.ignoreAll(Direction.IN_OUT);
    return ei;
  }

 public
  static EngineInterface dramRead(ManagerInterface m, LMemInterface iface) {
    m.addStreamToCPU("tocpu").connect(iface.addStreamFromLMem(
        "lmem2cpu", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
    EngineInterface ei = new EngineInterface("dramRead");
    CPUTypes TYPE = CPUTypes.INT;
    InterfaceParam size = ei.addParam("size_bytes", TYPE);
    InterfaceParam start = ei.addParam("start_bytes", TYPE);
    ei.setStream("tocpu", CPUTypes.UINT8, size);
    ei.setLMemLinear("lmem2cpu", start, size);
    ei.ignoreAll(Direction.IN_OUT);
    return ei;
  }

  //   // Enable debugging with Stream Status for the given manager
  //  public
  //   static void debug(ManagerInterface manager) {
  //     DebugLevel dbgLevel = new DebugLevel();
  //     dbgLevel.setHasStreamStatus(true);
  //     manager.debug.setDebugLevel(dbgLevel);
  //   }

 public
  static void addLinearStreamFromLMemToKernel(LMemInterface iface,
                                              KernelBlock kernel, String name) {
    kernel.getInput(name).connect(iface.addStreamFromLMem(
        name, LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
  }

 public
  static void addLinearStreamFromLMemToKernel(LMemCommandGroup group,
                                              KernelBlock kernel, String name) {
    kernel.getInput(name).connect(group.addStreamFromLMem(name));
  }

 public
  static void addLinearStreamFromLMemToKernel(LMemInterface iface,
                                              KernelBlock kernel, String src,
                                              String dst) {
    kernel.getInput(dst).connect(iface.addStreamFromLMem(
        src, LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
  }

 public
  static void addLinearStreamFromLMemToKernel(LMemCommandGroup group,
                                              KernelBlock kernel, String src,
                                              String dst) {
    kernel.getInput(dst).connect(group.addStreamFromLMem(src));
  }

 public
  static void addLinearStreamFromKernelToLMem(LMemInterface iface,
                                              KernelBlock kernel, String name) {
    iface.addStreamToLMem(name, LMemCommandGroup.MemoryAccessPattern.LINEAR_1D)
        .connect(kernel.getOutput(name));
  }

 public
  static void addLinearStreamFromKernelToLMem(LMemCommandGroup group,
                                              KernelBlock kernel, String name) {
    group.addStreamToLMem(name).connect(kernel.getOutput(name));
  }

 public
  static void addLinearStreamFromKernelToLMem(LMemInterface iface,
                                              KernelBlock kernel, String src,
                                              String dst) {
    iface.addStreamToLMem(dst, LMemCommandGroup.MemoryAccessPattern.LINEAR_1D)
        .connect(kernel.getOutput(src));
  }

 public
  static void addLinearStreamFromKernelToLMem(LMemCommandGroup group,
                                              KernelBlock kernel, String src,
                                              String dst) {
    group.addStreamToLMem(dst).connect(kernel.getOutput(src));
  }

 public
  static void ignoreLMemStreams(EngineInterface ei) {
    ei.ignoreLMem("cpu2lmem");
    ei.ignoreStream("fromcpu");
    ei.ignoreStream("tocpu");
    ei.ignoreLMem("lmem2cpu");
  }
}
