package com.custom_computing_ic.maxdeep.manager;

import com.custom_computing_ic.dfe_snippets.kernels.PaddingKernel;
import com.custom_computing_ic.dfe_snippets.kernels.UnpaddingKernel;
import com.custom_computing_ic.dfe_snippets.manager.ManagerUtils;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerWrapKernel;
import com.custom_computing_ic.maxdeep.kernel.conv2d.winograd.WinogradTransform;
import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.maxeler.maxcompiler.v2.managers.custom.DFELink;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.Demux;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.Fanout;
// import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.Mux;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemCommandGroup;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceMath;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manager related utilities.
 *
 * @author rz3515
 */
public class ConvLayerManagerUtils {
  public static final String IFMAP_NAME = "ifmap";
  public static final String OFMAP_NAME = "ofmap";
  public static final String COEFF_PREFIX = "coeff";
  public static final String DEPTHWISE_COEFF_PREFIX = "depthwise_coeff";
  public static final String POINTWISE_COEFF_PREFIX = "pointwise_coeff";

  public static InterfaceParam getBurstAlignedNumElems(
      InterfaceParam numElems, int sizeInBytes, InterfaceParam burstFactor, EngineInterface ei) {
    return getBurstAlignedNumElems(numElems, sizeInBytes, burstFactor, ei, 384);
  }

  public static InterfaceParam getBurstAlignedNumElems(InterfaceParam numElems, int sizeInBytes,
      InterfaceParam burstFactor, EngineInterface ei, int rawBurstSize) {
    InterfaceParam burstSize = ei.addConstant(rawBurstSize) * burstFactor;
    burstSize = burstSize.cast(CPUTypes.INT64);

    InterfaceParam totalSizeInBytes = numElems * sizeInBytes;

    InterfaceParam burstAlignedTotalSizeInBytes =
        InterfaceMath.ceil(totalSizeInBytes.cast(CPUTypes.DOUBLE) / burstSize.cast(CPUTypes.DOUBLE))
            .cast(CPUTypes.INT64)
        * burstSize;

    InterfaceParam burstAlignedNumElems = burstAlignedTotalSizeInBytes / sizeInBytes;

    return burstAlignedNumElems;
  }

  @SuppressWarnings("unused")
  private static boolean hasBNN(List<ConvLayerParameters> cps) {
    for (ConvLayerParameters cp : cps)
      if (cp.BW == 1)
        return true;
    return false;
  }

  public static Map<String, KernelBlock> createKernelBlocks(
      ManagerInterface mgr, ConvLayerParameters cp, boolean useDRAM) {
    return createKernelBlocks(mgr, cp, 1, useDRAM);
  }

  public static Map<String, KernelBlock> createKernelBlocks(
      ManagerInterface mgr, ConvLayerParameters cp, int numCoeffFifoSplits, boolean useDRAM) {
    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();
    cps.add(cp);

    return createKernelBlocks(mgr, cps, numCoeffFifoSplits, useDRAM);
  }

  public static boolean isInitCoeff(List<ConvLayerParameters> cps) {
    for (int i = 0; i < cps.size(); i++)
      if (cps.get(i).initCoeff)
        return true;
    return false;
  }

  /**
   * Create kernel blocks and connect them to streams
   *
   * TODO: should not only accept {@link ConvLayerParameters}
   *
   * @param mgr
   * @param cps
   * @return
   */
  public static Map<String, KernelBlock> createKernelBlocks(ManagerInterface mgr,
      List<ConvLayerParameters> cps, int numCoeffFifoSplits, boolean useDRAM) {
    Map<String, KernelBlock> knls = new HashMap<String, KernelBlock>();

    // create computing blocks
    for (ConvLayerParameters cp : cps) {
      String knlName = getKernelName(cp);

      mgr.logMsg("Generating kernel %s ...", knlName);
      KernelBlock knl = mgr.addKernel(
          new ConvLayerWrapKernel(mgr.makeKernelParameters(knlName), cp, numCoeffFifoSplits));
      knls.put(knlName, knl);
    }

    // create padding blocks
    Map<String, KernelBlock> padKnls = null;
    if (useDRAM) {
      mgr.logMsg("Generating padding kernels for DRAM access");
      padKnls = createPaddingKernels(mgr, cps, numCoeffFifoSplits);
    }

    KernelBlock knl = null;

    // check if we need to do fanout.
    Map<String, Fanout> fanouts = new HashMap<String, Fanout>();
    for (int i = 0; i < cps.size(); i++) {
      ConvLayerParameters cp = cps.get(i);
      if (!cp.residual.isEmpty()) {
        mgr.logMsg("Creating fanout with key: %s\n", cp.residual);
        fanouts.put(cp.residual, mgr.fanout(cp.residual + "_fanout"));
      }
    }

    boolean initCoeff = false;
    for (int i = 0; i < cps.size(); i++) initCoeff = initCoeff || cps.get(i).initCoeff;

    Demux initCoeffDemux = null;
    Mux initCoeffMux = null;
    if (initCoeff) {
      initCoeffDemux = mgr.demux("init_coeff_demux");
      initCoeffMux = mgr.mux("init_coeff_mux");

      initCoeffDemux.setClock(mgr.generateStaticClock("init_coeff_demux_clk", 50));
      initCoeffMux.setClock(mgr.generateStaticClock("init_coeff_mux_clk", 50));

      DFELink fromCPU = mgr.addStreamFromCPU("init_coeff");
      initCoeffDemux.getInput().connect(fromCPU);
      DFELink toCPU = mgr.addStreamToCPU("init_coeff_out");
      toCPU.connect(initCoeffMux.getOutput());
    }

    // setting up streams
    for (int i = 0; i < cps.size(); i++) {
      ConvLayerParameters cp = cps.get(i);
      knl = knls.get(getKernelName(cp));

      mgr.logMsg("Setting up stream connections for %s", getKernelName(cp));

      // connect ifmap
      if (i == 0) {
        if (useDRAM) {
          KernelBlock ifmapUnpadKnl = padKnls.get(getIfmapUnpadKernelName());

          // if the current layer has fanout input -
          if (fanouts.containsKey(cp.name)) {
            DFELink link = fanouts.get(cp.name).addOutput(cp.name);
            // connect ifmap to the output of unpad kernel
            knl.getInput(ConvLayerWrapKernel.IFMAP_NAME).connect(link);
            fanouts.get(cp.name).getInput().connect(
                ifmapUnpadKnl.getOutput(UnpaddingKernel.OUT_NAME));
          } else {
            // connect ifmap to the output of unpad kernel
            knl.getInput(ConvLayerWrapKernel.IFMAP_NAME)
                .connect(ifmapUnpadKnl.getOutput(UnpaddingKernel.OUT_NAME));
          }

        } else {
          if (fanouts.containsKey(cp.name))
            throw new UnsupportedOperationException("not implemented");
          knl.getInput(ConvLayerWrapKernel.IFMAP_NAME).connect(mgr.addStreamFromCPU(IFMAP_NAME));
        }
      } else {
        DFELink link;
        String key = cps.get(i - 1).name;
        if (fanouts.containsKey(cp.name))
          throw new UnsupportedOperationException("not implemented");
        // if (fanouts.containsKey(key))
        // link = fanouts.get(key).addOutput(cp.name);
        // else
        link = knls.get(key).getOutput(ConvLayerWrapKernel.OFMAP_NAME);

        knl.getInput(ConvLayerWrapKernel.IFMAP_NAME).connect(link);
      }

      // if (fanouts.containsKey(cp.name)) {
      //   fanouts.get(cp.name).getInput().connect(knl.getOutput(ConvLayerWrapKernel.OFMAP_NAME));
      // }

      // residual connection.
      if (!cp.residual.isEmpty()) {
        DFELink link;
        String key = cp.residual;
        if (fanouts.containsKey(key)) {
          link = fanouts.get(key).addOutput(cp.name);
          mgr.logMsg("Read from fanout: %s", key + "_fanout");
        } else
          link = knls.get(key).getOutput(ConvLayerWrapKernel.OFMAP_NAME);

        mgr.logMsg("Connecting to %s\n", link.getName());
        knl.getInput(ConvLayerWrapKernel.RESIDUAL_NAME).connect(link);
      }

      // connect coeff
      if (!cp.coeffOnChip) {
        if (cp.type == Type.STANDARD || cp.type == Type.DEPTHWISE_SEPARABLE_V2) {
          if (useDRAM) {
            KernelBlock coeffUnpadKnl = padKnls.get(getCoeffUnpadKernelName(cp));
            if (numCoeffFifoSplits == 1) {
              mgr.logMsg("There is only one coefficient stream in the design");
              knl.getInput(ConvLayerWrapKernel.COEFF_NAME)
                  .connect(coeffUnpadKnl.getOutput(UnpaddingKernel.OUT_NAME));
            } else {
              for (int s = 0; s < numCoeffFifoSplits; s++) {
                String coeffStrName = ConvLayerWrapKernel.COEFF_NAME + "_" + s;
                String unpadStrName = UnpaddingKernel.OUT_NAME + "_" + s;

                mgr.logMsg(String.format("Connecting coefficient split FIFO from %s to %s.",
                    unpadStrName, coeffStrName));

                knl.getInput(coeffStrName).connect(coeffUnpadKnl.getOutput(unpadStrName));
              }
            }
          } else {
            knl.getInput(ConvLayerWrapKernel.COEFF_NAME)
                .connect(mgr.addStreamFromCPU(COEFF_PREFIX + "_" + i));
          }

        } else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
          if (useDRAM) {
            KernelBlock depthCoeffUnpadKnl = padKnls.get(getDepthwiseCoeffUnpadKernelName(cp));
            knl.getInput(ConvLayerWrapKernel.DEPTHWISE_COEFF_NAME)
                .connect(depthCoeffUnpadKnl.getOutput(UnpaddingKernel.OUT_NAME));

            KernelBlock pointCoeffUnpadKnl = padKnls.get(getPointwiseCoeffUnpadKernelName(cp));
            knl.getInput(ConvLayerWrapKernel.POINTWISE_COEFF_NAME)
                .connect(pointCoeffUnpadKnl.getOutput(UnpaddingKernel.OUT_NAME));

          } else {
            knl.getInput(ConvLayerWrapKernel.DEPTHWISE_COEFF_NAME)
                .connect(mgr.addStreamFromCPU(DEPTHWISE_COEFF_PREFIX + "_" + i));
            knl.getInput(ConvLayerWrapKernel.POINTWISE_COEFF_NAME)
                .connect(mgr.addStreamFromCPU(POINTWISE_COEFF_PREFIX + "_" + i));
          }

        } else {
          throw new IllegalArgumentException("type is not supported");
        }

        // knl.getInput(ConvLayerWrapKernel.COEFF_NAME).connect(
        // mgr.addStreamFromCPU(COEFF_PREFIX + "_" + i));
      } else {
        if (cp.initCoeff) {
          knl.getInput(ConvLayerWrapKernel.INIT_COEFF_STREAM_NAME)
              .connect(initCoeffDemux.addOutput(cp.name));
          // knl.getInput(ConvLayerWrapKernel.INIT_COEFF_STREAM_NAME)
          //     .connect(mgr.addStreamFromCPU(cp.name + "_init_coeff"));

          initCoeffMux.addInput(cp.name).connect(
              knl.getOutput(ConvLayerWrapKernel.INIT_COEFF_STREAM_OUT_NAME));
        }
      }
    }

    if (useDRAM) {
      KernelBlock ofmapPadKnl = padKnls.get(getOfmapPadKernelName());
      ofmapPadKnl.getInput(PaddingKernel.INP_NAME).connect(knl.getOutput(OFMAP_NAME));
    } else {
      mgr.addStreamToCPU(OFMAP_NAME).connect(knl.getOutput(OFMAP_NAME));
    }

    return knls;
  }

  /**
   * Create padding and unpadding kernel blocks
   *
   * @param mgr
   * @param cps
   */
  public static Map<String, KernelBlock> createPaddingKernels(
      ManagerInterface mgr, List<ConvLayerParameters> cps, int numCoeffFifoSplits) {
    Map<String, KernelBlock> knls = new HashMap<String, KernelBlock>();

    // create unpadding kernels for coefficients
    for (int i = 0; i < cps.size(); i++) {
      ConvLayerParameters cp = cps.get(i);

      if (cp.coeffOnChip)
        continue;
      if (cp.type == Type.STANDARD || cp.type == Type.DEPTHWISE_SEPARABLE_V2) {
        String knlName = getCoeffUnpadKernelName(cp);
        KernelBlock knl = mgr.addKernel(new UnpaddingKernel(mgr.makeKernelParameters(knlName),
            cp.getCoeffStreamBitWidth() / cp.getCoeffStreamChunkSize(), numCoeffFifoSplits,
            cp.dbg));
        knls.put(knlName, knl);

      } else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
        String depthwiseKnlName = getDepthwiseCoeffUnpadKernelName(cp);
        String pointwiseKnlName = getPointwiseCoeffUnpadKernelName(cp);

        KernelBlock depthwiseKnl =
            mgr.addKernel(new UnpaddingKernel(mgr.makeKernelParameters(depthwiseKnlName),
                cp.getDepthwiseCoeffStreamBitWidth() / (cp.K * cp.K), cp.dbg));
        knls.put(depthwiseKnlName, depthwiseKnl);

        KernelBlock pointwiseKnl =
            mgr.addKernel(new UnpaddingKernel(mgr.makeKernelParameters(pointwiseKnlName),
                cp.getPointwiseCoeffStreamBitWidth(), cp.dbg));
        knls.put(pointwiseKnlName, pointwiseKnl);

      } else {
        throw new IllegalArgumentException("type is not supported");
      }
    }

    ConvLayerParameters cpf = cps.get(0);
    ConvLayerParameters cpl = cps.get(cps.size() - 1);

    String ifmapUnpadKnlName = getIfmapUnpadKernelName();
    KernelBlock ifmapUnpadKnl = mgr.addKernel(new UnpaddingKernel(
        mgr.makeKernelParameters(ifmapUnpadKnlName), cpf.getIfmapStreamBitWidth(), cpf.dbg));
    knls.put(ifmapUnpadKnlName, ifmapUnpadKnl);

    String ofmapPadKnlName = getOfmapPadKernelName();
    KernelBlock ofmapPadKnl = mgr.addKernel(new PaddingKernel(
        mgr.makeKernelParameters(ofmapPadKnlName), cpl.getOfmapStreamBitWidth(), cpl.dbg));
    knls.put(ofmapPadKnlName, ofmapPadKnl);

    // setup connections to the LMem
    LMemInterface iface = mgr.getLMemInterface();
    LMemCommandGroup groupIfmap =
        iface.addCommandGroup("GROUP_IFMAP", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);
    LMemCommandGroup groupOfmap =
        iface.addCommandGroup("GROUP_OFMAP", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);

    for (int i = 0; i < cps.size(); i++) {
      ConvLayerParameters cp = cps.get(i);

      if (cp.coeffOnChip)
        continue;

      if (cp.type == Type.STANDARD || cp.type == Type.DEPTHWISE_SEPARABLE_V2) {
        LMemCommandGroup groupCoeff = iface.addCommandGroup("GROUP_COEFF"
                + "_" + i,
            LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);
        String coeffUnpadKnlName = getCoeffUnpadKernelName(cp);
        KernelBlock coeffUnpadKnl = knls.get(coeffUnpadKnlName);

        ManagerUtils.addLinearStreamFromLMemToKernel(
            groupCoeff, coeffUnpadKnl, COEFF_PREFIX + "_" + i, UnpaddingKernel.INP_NAME);

      } else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
        LMemCommandGroup groupDepthCoeff = iface.addCommandGroup("GROUP_DEPTH_COEFF"
                + "_" + i,
            LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);
        String depthwiseKnlName = getDepthwiseCoeffUnpadKernelName(cp);
        KernelBlock depthwiseCoeffUnpadKnl = knls.get(depthwiseKnlName);

        ManagerUtils.addLinearStreamFromLMemToKernel(groupDepthCoeff, depthwiseCoeffUnpadKnl,
            DEPTHWISE_COEFF_PREFIX + "_" + i, UnpaddingKernel.INP_NAME);

        LMemCommandGroup groupPointCoeff = iface.addCommandGroup("GROUP_POINT_COEFF"
                + "_" + i,
            LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);
        String pointwiseKnlName = getPointwiseCoeffUnpadKernelName(cp);
        KernelBlock pointwiseCoeffUnpadKnl = knls.get(pointwiseKnlName);

        ManagerUtils.addLinearStreamFromLMemToKernel(groupPointCoeff, pointwiseCoeffUnpadKnl,
            POINTWISE_COEFF_PREFIX + "_" + i, UnpaddingKernel.INP_NAME);
      } else {
        throw new IllegalArgumentException("type is not supported");
      }
    }

    ManagerUtils.addLinearStreamFromLMemToKernel(
        groupIfmap, ifmapUnpadKnl, IFMAP_NAME, UnpaddingKernel.INP_NAME);
    ManagerUtils.addLinearStreamFromKernelToLMem(
        groupOfmap, ofmapPadKnl, PaddingKernel.OUT_NAME, OFMAP_NAME);

    return knls;
  }

  public static String getDepthwiseCoeffUnpadKernelName(ConvLayerParameters cp) {
    return String.format("%s_depthwise_coeff_unpad", getKernelName(cp));
  }

  public static String getPointwiseCoeffUnpadKernelName(ConvLayerParameters cp) {
    return String.format("%s_pointwise_coeff_unpad", getKernelName(cp));
  }

  public static String getCoeffUnpadKernelName(ConvLayerParameters cp) {
    return String.format("%s_coeff_unpad", getKernelName(cp));
  }

  public static String getIfmapUnpadKernelName() {
    return "ifmap_unpad";
  }

  public static String getOfmapPadKernelName() {
    return "ofmap_pad";
  }

  public static void setupStreams(
      EngineInterface ei, ConvLayerParameters cp, InterfaceParam batchSize, boolean useDRAM) {
    setupStreams(ei, cp, batchSize, useDRAM, null);
  }

  public static void setupStreams(EngineInterface ei, ConvLayerParameters cp,
      InterfaceParam batchSize, boolean useDRAM, ManagerInterface mgr) {
    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();
    cps.add(cp);

    setupStreams(ei, cps, batchSize, useDRAM, mgr);
  }

  public static void setupStreams(EngineInterface ei, List<ConvLayerParameters> cps,
      InterfaceParam batchSize, boolean useDRAM) {
    setupStreams(ei, cps, batchSize, useDRAM, null);
  }

  /**
   * Set stream size in the engine interface
   *
   * @param ei
   * @param cps
   * @param batchSize
   */
  public static void setupStreams(EngineInterface ei, List<ConvLayerParameters> cps,
      InterfaceParam batchSize, boolean useDRAM, ManagerInterface mgr) {
    if (isInitCoeff(cps)) {
      ei.ignoreRoute("init_coeff_demux");
      ei.ignoreRoute("init_coeff_mux");
    }

    if (useDRAM) {
      if (mgr != null)
        mgr.logMsg("DRAM will be used to build the design");

      // base address of the memory space, will be updated once
      // a new block is allocated.
      InterfaceParam baseAddr;

      // setup the stream input from ifmap stored in DRAM
      baseAddr = setupIfmapStream(ei, cps.get(0), batchSize);

      // for each convolution layer to be built
      for (int i = 0; i < cps.size(); i++) {
        ConvLayerParameters cp = cps.get(i);
        if (cp.initCoeff)
          ei.setScalar(cp.name, ConvLayerWrapKernel.INIT_COEFF_NAME, 0);

        ei.setTicks(getKernelName(cp), cp.getNumCycles() * batchSize);

        if (mgr != null) {
          mgr.logMsg("Setup streams for kernel \"%s\"", cp.name);
          mgr.logMsg("# cycles:       %d%n", cp.getNumCycles());
          mgr.logMsg("# ifmap stream: %d", cp.getIfmapStreamNumElems());
          mgr.logMsg("# coeff stream: %d", cp.getCoeffStreamNumElems());
          mgr.logMsg("# ofmap stream: %d", cp.getOfmapStreamNumElems());
          mgr.logMsg("coeff vec size: %d", cp.getCoeffVecSize());
          mgr.logMsg("coeff stream bit width: %d", cp.getCoeffStreamBitWidth());
          mgr.logMsg("coeff stream chunk size: %d", cp.getCoeffStreamChunkSize());
        }

        if (!cp.coeffOnChip) {
          if (cp.type == Type.STANDARD || cp.type == Type.DEPTHWISE_SEPARABLE_V2) {
            baseAddr = setupCoeffStream(ei, cp, i, batchSize, baseAddr);
          } else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
            baseAddr = setupCoeffStream(ei, cps.get(i), i, batchSize, baseAddr, true);
            baseAddr = setupCoeffStream(ei, cps.get(i), i, batchSize, baseAddr, false);
          } else {
            throw new IllegalArgumentException("type is not supported");
          }
        } else {
          if (cp.initCoeff) {
            ei.ignoreStream("init_coeff");
            ei.ignoreStream("init_coeff_out");
          }
        }
      }

      baseAddr = setupOfmapStream(ei, cps.get(cps.size() - 1), batchSize, baseAddr);

    } else {
      for (int i = 0; i < cps.size(); i++) {
        ConvLayerParameters cp = cps.get(i);

        if (mgr != null) {
          mgr.logMsg("Setup streams for kernel \"%s\"", cp.name);
          mgr.logMsg("# cycles per batch: %d", cp.getNumCycles());
          mgr.logMsg("# ifmap stream: %d", cp.getIfmapStreamNumElems());
          mgr.logMsg("# coeff stream: %d", cp.getCoeffStreamNumElems());
          mgr.logMsg("# ofmap stream: %d", cp.getOfmapStreamNumElems());
          mgr.logMsg("coeff vec size: %d", cp.getCoeffVecSize());
        }

        ei.setTicks(getKernelName(cp), cp.getNumCycles() * batchSize);

        if (cp.type == Type.STANDARD || cp.type == Type.DEPTHWISE_SEPARABLE_V2)
          ei.setStream(
              COEFF_PREFIX + "_" + i, cp.getCPUTypes(), cp.getCoeffStreamSize() * batchSize);
        else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
          ei.setStream(DEPTHWISE_COEFF_PREFIX + "_" + i, cp.getCPUTypes(),
              cp.getDepthwiseCoeffStreamSize() * batchSize);
          ei.setStream(POINTWISE_COEFF_PREFIX + "_" + i, cp.getCPUTypes(),
              cp.getPointwiseCoeffStreamSize() * batchSize);
        } else {
          throw new IllegalArgumentException("type is not supported");
        }
      }

      ConvLayerParameters cpf = cps.get(0);
      ConvLayerParameters cpl = cps.get(cps.size() - 1);

      System.out.printf("ifmap size = %d\n", cpf.getIfmapStreamSize());

      ei.setStream(IFMAP_NAME, cpf.getCPUTypes(), cpf.getIfmapStreamSize() * batchSize);
      ei.setStream(OFMAP_NAME, cpl.getCPUTypes(), cpl.getOfmapStreamSize() * batchSize);
    }
  }

  public static InterfaceParam setupIfmapStream(
      EngineInterface ei, ConvLayerParameters cp, InterfaceParam batchSize) {
    String ifmapUnpadKnlName = getIfmapUnpadKernelName();
    String LMemStreamName = IFMAP_NAME;

    return setupIfmapStream(ei, cp, batchSize, LMemStreamName, ifmapUnpadKnlName);
  }

  public static InterfaceParam setupIfmapStream(EngineInterface ei, ConvLayerParameters cp,
      InterfaceParam batchSize, String LMemStreamName, String ifmapUnpadKnlName) {
    InterfaceParam ifmapNumElems = ei.addConstant(cp.getIfmapStreamNumElems()).cast(CPUTypes.INT64);
    ifmapNumElems *= batchSize;

    InterfaceParam burstFactor = ei.addConstant(cp.getIfmapVecSize()).cast(CPUTypes.INT64);
    InterfaceParam burstAlignedIfmapNumElems =
        getBurstAlignedNumElems(ifmapNumElems, cp.getCPUTypes().sizeInBytes(), burstFactor, ei);
    InterfaceParam burstAlignedIfmapSize =
        burstAlignedIfmapNumElems * cp.getCPUTypes().sizeInBytes();

    ei.setScalar(ifmapUnpadKnlName, UnpaddingKernel.SCALAR_NUM_INP, ifmapNumElems / burstFactor);
    ei.setScalar(ifmapUnpadKnlName, UnpaddingKernel.SCALAR_TOTAL_CYCLES,
        burstAlignedIfmapNumElems / burstFactor);

    ei.setTicks(ifmapUnpadKnlName, burstAlignedIfmapNumElems / burstFactor);

    InterfaceParam ZERO = ei.addConstant(0).cast(CPUTypes.INT64);
    ei.setLMemLinear(LMemStreamName, ZERO, burstAlignedIfmapSize);

    return burstAlignedIfmapSize;
  }

  public static InterfaceParam setupCoeffStream(EngineInterface ei, ConvLayerParameters cp,
      int index, InterfaceParam batchSize, InterfaceParam baseAddr, boolean isDepthwise) {
    String knlName =
        (isDepthwise) ? getDepthwiseCoeffUnpadKernelName(cp) : getPointwiseCoeffUnpadKernelName(cp);
    long numElemsValue =
        (isDepthwise) ? cp.getDepthwiseCoeffStreamNumElems() : cp.getPointwiseCoeffStreamNumElems();
    InterfaceParam numElems = ei.addConstant(numElemsValue).cast(CPUTypes.INT64);
    numElems *= batchSize;

    int burstFactorSize = (isDepthwise) ? (cp.PC) : (cp.PC * cp.PF);
    InterfaceParam burstFactor = ei.addConstant(burstFactorSize).cast(CPUTypes.INT64);

    InterfaceParam burstAlignedNumElems =
        getBurstAlignedNumElems(numElems, cp.getCPUTypes().sizeInBytes(), burstFactor, ei);
    InterfaceParam burstAlignedSize = burstAlignedNumElems * cp.getCPUTypes().sizeInBytes();

    ei.setScalar(knlName, UnpaddingKernel.SCALAR_NUM_INP, numElems / burstFactor);
    ei.setScalar(knlName, UnpaddingKernel.SCALAR_TOTAL_CYCLES, burstAlignedNumElems / burstFactor);

    InterfaceParam numTicks =
        (isDepthwise) ? (burstAlignedNumElems / burstFactor) : (burstAlignedNumElems / burstFactor);
    ei.setTicks(knlName, numTicks);

    String prefix = (isDepthwise) ? DEPTHWISE_COEFF_PREFIX : POINTWISE_COEFF_PREFIX;
    ei.setLMemLinear(prefix + "_" + index, baseAddr, burstAlignedSize);

    return baseAddr + burstAlignedSize;
  }

  public static InterfaceParam setupCoeffStream(EngineInterface ei, ConvLayerParameters cp,
      int index, InterfaceParam batchSize, InterfaceParam baseAddr) {
    String knlName = getCoeffUnpadKernelName(cp);
    InterfaceParam numElems = ei.addConstant(cp.getCoeffStreamNumElems()).cast(CPUTypes.INT64);
    numElems *= batchSize;

    InterfaceParam burstFactor =
        ei.addConstant(cp.getCoeffVecSize() / cp.getCoeffStreamChunkSize()).cast(CPUTypes.INT64);
    InterfaceParam burstAlignedNumElems =
        getBurstAlignedNumElems(numElems, cp.getCPUTypes().sizeInBytes(), burstFactor, ei);
    InterfaceParam burstAlignedSize = burstAlignedNumElems * cp.getCPUTypes().sizeInBytes();

    ei.setScalar(knlName, UnpaddingKernel.SCALAR_NUM_INP, numElems / burstFactor);
    ei.setScalar(knlName, UnpaddingKernel.SCALAR_TOTAL_CYCLES, burstAlignedNumElems / burstFactor);

    ei.setTicks(knlName, burstAlignedNumElems / burstFactor);

    ei.setLMemLinear(COEFF_PREFIX + "_" + index, baseAddr, burstAlignedSize);

    return baseAddr + burstAlignedSize;
  }

  public static InterfaceParam setupOfmapStream(EngineInterface ei, ConvLayerParameters cp,
      InterfaceParam batchSize, InterfaceParam baseAddr) {
    String knlName = getOfmapPadKernelName();
    InterfaceParam numElems = ei.addConstant(cp.getOfmapStreamNumElems()).cast(CPUTypes.INT64);
    numElems *= batchSize;

    InterfaceParam burstFactor = ei.addConstant(cp.getOfmapVecSize()).cast(CPUTypes.INT64);
    InterfaceParam burstAlignedNumElems =
        getBurstAlignedNumElems(numElems, cp.getCPUTypes().sizeInBytes(), burstFactor, ei);
    InterfaceParam burstAlignedSize = burstAlignedNumElems * cp.getCPUTypes().sizeInBytes();

    ei.setScalar(knlName, PaddingKernel.SCALAR_NUM_INP, numElems / burstFactor);
    ei.setScalar(knlName, PaddingKernel.SCALAR_TOTAL_CYCLES, burstAlignedNumElems / burstFactor);

    ei.setTicks(knlName, burstAlignedNumElems / burstFactor);

    ei.setLMemLinear(OFMAP_NAME, baseAddr, burstAlignedSize);

    return baseAddr + burstAlignedNumElems;
  }

  public static EngineInterface initCoeff(
      List<ConvLayerParameters> cps, ConvLayerEngineParameters ep) {
    EngineInterface ei = new EngineInterface("initCoeff");

    ei.ignoreKernel(getIfmapUnpadKernelName());
    ei.ignoreKernel(getOfmapPadKernelName());
    ei.setTicks(getIfmapUnpadKernelName(), 0);
    ei.setTicks(getOfmapPadKernelName(), 0);

    ManagerUtils.ignoreLMemStreams(ei);
    ei.ignoreLMem(IFMAP_NAME);
    ei.ignoreLMem(OFMAP_NAME);

    InterfaceParam idx = ei.addParam("init_coeff_kernel_index", CPUTypes.INT);
    InterfaceParam N =
        ei.addParam("init_coeff_num_elems", CPUTypes.UINT64, "Number of coefficient elements");
    ei.setStream("init_coeff", CPUTypes.INT8, N.mul(CPUTypes.INT8.sizeInBytes()));
    ei.setStream("init_coeff_out", CPUTypes.INT8, N.mul(CPUTypes.INT8.sizeInBytes()));
    // ei.setStream("conv0_init_coeff", CPUTypes.INT8, N.mul(CPUTypes.INT8.sizeInBytes()));
    // ei.setStream("conv0_init_coeff_out", CPUTypes.INT8, N.mul(CPUTypes.INT8.sizeInBytes()));
    // ei.ignoreStream("init_coeff");
    // ei.ignoreStream("conv1_init_coeff");
    // ei.ignoreStream("conv2_init_coeff");
    // ei.ignoreStream("conv1_init_coeff_out");
    // ei.ignoreStream("conv2_init_coeff_out");
    // ei.ignoreRoute("init_coeff_demux");

    for (int i = 0; i < cps.size(); ++i) {
      ConvLayerParameters cp = cps.get(i);

      if (cp.F % cp.PF != 0)
        throw new IllegalArgumentException("F % PF == 0");
      if (cp.C % cp.PC != 0)
        throw new IllegalArgumentException("C % PC == 0");

      // ei.setScalar(cp.name, ConvLayerWrapKernel.INIT_COEFF_NAME, idx.eq(i));
      // ei.setTicks(cp.name, idx.eq(i).cast(CPUTypes.INT).mul(cp.F * cp.C * cp.K * cp.K));

      if (!cp.residual.isEmpty())
        ei.ignoreRoute(cp.residual + "_fanout");
    }

    return ei;
  }

  public static void setupConstants(
      ManagerInterface mgr, ConvLayerParameters cp, ConvLayerEngineParameters ep) {
    List<ConvLayerParameters> cps = new ArrayList<ConvLayerParameters>();
    cps.add(cp);

    setupConstants(mgr, cps, ep);
  }

  public static void setupConstants(
      ManagerInterface mgr, List<ConvLayerParameters> cps, ConvLayerEngineParameters ep) {
    // setup the definition of constants of the currrent hardware build
    for (int i = 0; i < cps.size(); i++) {
      ConvLayerParameters cp = cps.get(i);

      String name = cp.name;

      mgr.addMaxFileConstant(name + "_H", cp.H);
      mgr.addMaxFileConstant(name + "_W", cp.W);
      mgr.addMaxFileConstant(name + "_OH", cp.OH);
      mgr.addMaxFileConstant(name + "_OW", cp.OW);
      mgr.addMaxFileConstant(name + "_C", cp.C);
      mgr.addMaxFileConstant(name + "_F", cp.F);
      mgr.addMaxFileConstant(name + "_K", cp.K);
      mgr.addMaxFileConstant(name + "_PAD", cp.PAD);
      mgr.addMaxFileConstant(name + "_STRIDE", cp.STRIDE);
      mgr.addMaxFileConstant(name + "_BW", cp.BW);
      mgr.addMaxFileConstant(name + "_num_frac_bits", cp.numFracBits);
      mgr.addMaxFileStringConstant(name + "_dtype", cp.dtype);
      mgr.addMaxFileConstant(name + "_COEFF_ON_CHIP", cp.coeffOnChip ? 1 : 0);
      mgr.addMaxFileConstant(name + "_PC", cp.PC);
      mgr.addMaxFileConstant(name + "_PF", cp.PF);
      mgr.addMaxFileConstant(name + "_PK", cp.PK);
    }

    mgr.addMaxFileConstant("USE_DRAM", ep.getUseDRAM() ? 1 : 0);
    mgr.addMaxFileConstant("USE_WINO", ep.getUseWinograd() ? 1 : 0);
    mgr.addMaxFileConstant("WINO_TILE_SIZE", WinogradTransform.TILE_SIZE);
    mgr.addMaxFileConstant("WINO_M", WinogradTransform.M);
    mgr.addMaxFileConstant("WINO_COEFF_OFFLINE", ep.getWinogradWeightsOffline() ? 1 : 0);
  }

  public static String getKernelName(ConvLayerParameters cp) {
    return cp.name;
  }
}
