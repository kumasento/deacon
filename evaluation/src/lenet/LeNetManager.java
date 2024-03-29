package lenet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.custom_computing_ic.dfe_snippets.kernels.PaddingKernel;
import com.custom_computing_ic.dfe_snippets.kernels.UnpaddingKernel;
import com.custom_computing_ic.dfe_snippets.manager.ManagerUtils;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.CompSeq;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerWrapKernel;
import com.custom_computing_ic.maxdeep.kernel.fc.FullyConnectedLayerKernel;
import com.custom_computing_ic.maxdeep.kernel.fc.FullyConnectedLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.pool.PoolingLayerKernel;
import com.custom_computing_ic.maxdeep.kernel.pool.PoolingLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.pool.PoolingLayerParameters.Mode;
import com.custom_computing_ic.maxdeep.lib.LayerParameters;
import com.custom_computing_ic.maxdeep.manager.ConvLayerManagerUtils;
import com.maxeler.maxcompiler.v2.managers.BuildConfig;
import com.maxeler.maxcompiler.v2.managers.BuildConfig.Effort;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemCommandGroup;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;
import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;

public final class LeNetManager extends CustomManager {

  public LeNetManager(LeNetEngineParameters ep, List<LayerParameters> lps) {
    super(ep);

    config.setAllowNonMultipleTransitions(true);
    config.setDefaultStreamClockFrequency(ep.getFreq());

    logMsg("Start constructing LeNet accelerator design ...");

    /* create essential kernels for LeNet */
    logMsg("Start creating kernels ...");
    Map<String, KernelBlock> knls = createKernels(lps);

    /* create LMem groups */
    logMsg("Start creating LMem groups ...");
    Map<String, LMemCommandGroup> grps = createLMemGroups(lps);

    /* connect kernels */
    logMsg("Start connecting kernels ...");
    connectKernels(lps, knls, grps);
  }

  /**
   * Connect kernels
   * 
   * @param lps
   * @param knls
   */
  private void connectKernels(
      List<LayerParameters> lps,
      Map<String, KernelBlock> knls,
      Map<String, LMemCommandGroup> grps) {

    /* connect conv0 */
    logMsg("Start connecting kernel conv0");
    connectConvLayerKernel(
        (ConvLayerParameters) lps.get(0),
        knls,
        grps,
        null,
        false);

    /* connect conv1 */
    logMsg("Start connecting kernel conv1");
    connectConvLayerKernel(
        (ConvLayerParameters) lps.get(1),
        knls,
        grps,
        (ConvLayerParameters) lps.get(0),
        false);

    /* connect fc0 */
    logMsg("Start connecting kernel fc0");
    connectFullyConnectedLayerKernel(
        (FullyConnectedLayerParameters) lps.get(2),
        knls,
        grps,
        (ConvLayerParameters) lps.get(1),
        null,
        false);

    /* connect fc1 */
    logMsg("Start connecting kernel fc1");
    connectFullyConnectedLayerKernel(
        (FullyConnectedLayerParameters) lps.get(3),
        knls,
        grps,
        null,
        (FullyConnectedLayerParameters) lps.get(2),
        true);

    logMsg("Finished connection");
  }

  public static void connectConvLayerKernel(
      ConvLayerParameters cp,
      Map<String, KernelBlock> knls,
      Map<String, LMemCommandGroup> grps,
      ConvLayerParameters prevCp,
      boolean isLast) {
    /* ifmap */
    if (prevCp != null) {
      // connect to the ofmap of the pooling kernel if there is one
      if (prevCp.pool != null) {
        // conv <- prev pool
        knls
            .get(cp.name)
            .getInput(ConvLayerWrapKernel.IFMAP_NAME)
            .connect(
                knls.get(prevCp.name + "_pool").getOutput(
                    PoolingLayerKernel.OFMAP_NAME));
      } else {
        // conv <- prev conv
        knls
            .get(cp.name)
            .getInput(ConvLayerWrapKernel.IFMAP_NAME)
            .connect(
                knls.get(prevCp.name).getOutput(ConvLayerWrapKernel.OFMAP_NAME));
      }
    } else {
      // conv <- ifmap
      knls
          .get(cp.name)
          .getInput(ConvLayerWrapKernel.IFMAP_NAME)
          .connect(
              knls.get(cp.name + "_ifmap").getOutput(UnpaddingKernel.OUT_NAME));

      // ifmap <- LMem
      ManagerUtils.addLinearStreamFromLMemToKernel(
          grps.get("ifmap"),
          knls.get(cp.name + "_ifmap"),
          "LMEM_IFMAP",
          UnpaddingKernel.INP_NAME);
    }

    /* coeff */
    if (cp.type == Type.STANDARD) {
      // conv <- coeff
      knls
          .get(cp.name)
          .getInput(ConvLayerWrapKernel.COEFF_NAME)
          .connect(
              knls.get(cp.name + "_coeff").getOutput(UnpaddingKernel.OUT_NAME));
      // coeff <- LMem
      ManagerUtils.addLinearStreamFromLMemToKernel(
          grps.get(cp.name + "_coeff"),
          knls.get(cp.name + "_coeff"),
          cp.name + "_LMEM_COEFF",
          UnpaddingKernel.INP_NAME);
    } else {
      // conv <- depthwise
      knls
          .get(cp.name)
          .getInput(ConvLayerWrapKernel.DEPTHWISE_COEFF_NAME)
          .connect(
              knls.get(cp.name + "_depthwise_coeff").getOutput(
                  UnpaddingKernel.OUT_NAME));
      // depthwise <- LMem
      ManagerUtils.addLinearStreamFromLMemToKernel(
          grps.get(cp.name + "_depthwise_coeff"),
          knls.get(cp.name + "_depthwise_coeff"),
          cp.name + "_LMEM_DEPTHWISE_COEFF",
          UnpaddingKernel.INP_NAME);

      // conv <- pointwise
      knls
          .get(cp.name)
          .getInput(ConvLayerWrapKernel.POINTWISE_COEFF_NAME)
          .connect(
              knls.get(cp.name + "_pointwise_coeff").getOutput(
                  UnpaddingKernel.OUT_NAME));
      // pointwise <- LMem
      ManagerUtils.addLinearStreamFromLMemToKernel(
          grps.get(cp.name + "_pointwise_coeff"),
          knls.get(cp.name + "_pointwise_coeff"),
          cp.name + "_LMEM_POINTWISE_COEFF",
          UnpaddingKernel.INP_NAME);
    }

    /* pooling */
    if (cp.pool != null) {
      // pool <- conv
      knls
          .get(cp.name + "_pool")
          .getInput(PoolingLayerKernel.IFMAP_NAME)
          .connect(knls.get(cp.name).getOutput(ConvLayerWrapKernel.OFMAP_NAME));
    }

    /* ofmap */
    if (isLast) {
      if (cp.pool != null) {
        // ofmap <- pool
        knls
            .get(cp.name + "_ofmap")
            .getInput(PaddingKernel.INP_NAME)
            .connect(
                knls.get(cp.name + "_pool").getOutput(
                    PoolingLayerKernel.OFMAP_NAME));
      } else {
        // ofmap <- conv
        knls
            .get(cp.name + "_ofmap")
            .getInput(PaddingKernel.INP_NAME)
            .connect(
                knls.get(cp.name).getOutput(ConvLayerWrapKernel.OFMAP_NAME));
      }

      // ofmap
      ManagerUtils.addLinearStreamFromKernelToLMem(
          grps.get("ofmap"),
          knls.get(cp.name + "_ofmap"),
          PaddingKernel.OUT_NAME,
          "LMEM_OFMAP");
    }
  }

  public static void connectFullyConnectedLayerKernel(
      FullyConnectedLayerParameters fp,
      Map<String, KernelBlock> knls,
      Map<String, LMemCommandGroup> grps,
      ConvLayerParameters prevCp,
      FullyConnectedLayerParameters prevFp,
      boolean isLast) {

    if (prevCp != null && prevFp != null)
      throw new IllegalArgumentException("cannot have two previous layers");

    if (prevCp != null) {
      if (prevCp.pool != null) {
        // fc <- prev pool
        knls
            .get(fp.name)
            .getInput(FullyConnectedLayerKernel.IFMAP_NAME)
            .connect(
                knls.get(prevCp.name + "_pool").getOutput(
                    PoolingLayerKernel.OFMAP_NAME));
      } else {
        // fc <- prev conv
        knls
            .get(fp.name)
            .getInput(FullyConnectedLayerKernel.IFMAP_NAME)
            .connect(
                knls.get(prevCp.name).getOutput(ConvLayerWrapKernel.OFMAP_NAME));
      }
    } else if (prevFp != null) {
      // fc <- prev fc
      knls
          .get(fp.name)
          .getInput(FullyConnectedLayerKernel.IFMAP_NAME)
          .connect(
              knls.get(prevFp.name).getOutput(
                  FullyConnectedLayerKernel.OFMAP_NAME));
    } else {
      // fc <- ifmap
      knls
          .get(fp.name)
          .getInput(FullyConnectedLayerKernel.IFMAP_NAME)
          .connect(
              knls.get(fp.name + "_ifmap").getOutput(UnpaddingKernel.OUT_NAME));

      // ifmap <- LMem
      ManagerUtils.addLinearStreamFromLMemToKernel(
          grps.get("ifmap"),
          knls.get(fp.name + "_ifmap"),
          "LMEM_IFMAP",
          UnpaddingKernel.INP_NAME);
    }

    // fc <- coeff
    knls
        .get(fp.name)
        .getInput(FullyConnectedLayerKernel.COEFF_NAME)
        .connect(
            knls.get(fp.name + "_coeff").getOutput(UnpaddingKernel.OUT_NAME));
    // coeff <- LMem
    ManagerUtils.addLinearStreamFromLMemToKernel(
        grps.get(fp.name + "_coeff"),
        knls.get(fp.name + "_coeff"),
        fp.name + "_LMEM_COEFF",
        UnpaddingKernel.INP_NAME);

    /* ofmap */
    if (isLast) {
      // ofmap <- fc
      knls
          .get(fp.name + "_ofmap")
          .getInput(PaddingKernel.INP_NAME)
          .connect(
              knls.get(fp.name).getOutput(FullyConnectedLayerKernel.OFMAP_NAME));

      // ofmap
      ManagerUtils.addLinearStreamFromKernelToLMem(
          grps.get("ofmap"),
          knls.get(fp.name + "_ofmap"),
          PaddingKernel.OUT_NAME,
          "LMEM_OFMAP");
    }

  }

  /**
   * Create kernels belong to LeNet
   * 
   * @param lps
   * @return
   */
  private Map<String, KernelBlock> createKernels(List<LayerParameters> lps) {
    Map<String, KernelBlock> knls = new HashMap<String, KernelBlock>();

    /* cast parameters */
    ConvLayerParameters cp0 = (ConvLayerParameters) lps.get(0);
    ConvLayerParameters cp1 = (ConvLayerParameters) lps.get(1);
    FullyConnectedLayerParameters fp0 =
        (FullyConnectedLayerParameters) lps.get(2);
    FullyConnectedLayerParameters fp1 =
        (FullyConnectedLayerParameters) lps.get(3);

    /* conv0 */
    createConvLayerKernels(knls, cp0, true, false);
    /* conv1 */
    createConvLayerKernels(knls, cp1, false, false);
    /* fc0 */
    createFullyConnectedLayerKernels(knls, fp0, false, false);
    /* fc1 */
    createFullyConnectedLayerKernels(knls, fp1, false, true);

    return knls;
  }

  /**
   * Create fully connected layer kernels
   * 
   * @param knls
   * @param fp
   * @param isFirst
   * @param isLast
   */
  public void createFullyConnectedLayerKernels(
      Map<String, KernelBlock> knls,
      FullyConnectedLayerParameters fp,
      boolean isFirst,
      boolean isLast) {
    /* core */
    knls.put(fp.name, addKernel(new FullyConnectedLayerKernel(
        makeKernelParameters(fp.name),
        fp)));

    /* coeff */
    knls.put(fp.name + "_coeff", addKernel(new UnpaddingKernel(
        makeKernelParameters(fp.name + "_coeff"),
        fp.getCoeffStreamBitWidth(),
        false)));

    /* ifmap */
    if (isFirst)
      knls.put(fp.name + "_ifmap", addKernel(new UnpaddingKernel(
          makeKernelParameters(fp.name + "_ifmap"),
          fp.getIfmapStreamBitWidth(),
          false)));

    /* ofmap */
    if (isLast)
      knls.put(fp.name + "_ofmap", addKernel(new PaddingKernel(
          makeKernelParameters(fp.name + "_ofmap"),
          fp.getOfmapStreamBitWidth(),
          false)));
  }

  /**
   * create convolution layer kernels
   * 
   * @param knls
   * @param cp
   * @param isFirst
   * @param isLast
   */
  public void createConvLayerKernels(
      Map<String, KernelBlock> knls,
      ConvLayerParameters cp,
      boolean isFirst,
      boolean isLast) {
    /* core conv layer kernel */
    knls.put(cp.name, addKernel(new ConvLayerWrapKernel(
        makeKernelParameters(cp.name),
        cp)));

    /* coeff unpad kernels */
    if (cp.type == Type.DEPTHWISE_SEPARABLE) {
      knls.put(cp.name + "_depthwise_coeff", addKernel(new UnpaddingKernel(
          makeKernelParameters(cp.name + "_depthwise_coeff"),
          cp.getDepthwiseCoeffStreamLMemBitWidth(),
          cp.dbg)));
      knls.put(cp.name + "_pointwise_coeff", addKernel(new UnpaddingKernel(
          makeKernelParameters(cp.name + "_pointwise_coeff"),
          cp.getPointwiseCoeffStreamBitWidth(),
          cp.dbg)));
    } else
      knls.put(cp.name + "_coeff", addKernel(new UnpaddingKernel(
          makeKernelParameters(cp.name + "_coeff"),
          cp.getCoeffStreamLMemBitWidth(),
          cp.dbg)));

    /* pooling */
    if (cp.pool != null) {
      knls.put(cp.name + "_pool", addKernel(new PoolingLayerKernel(
          makeKernelParameters(cp.name + "_pool"),
          cp)));
    }

    /* input unpad kernel */
    if (isFirst)
      knls.put(cp.name + "_ifmap", addKernel(new UnpaddingKernel(
          makeKernelParameters(cp.name + "_ifmap"),
          cp.getIfmapStreamBitWidth(),
          cp.dbg)));

    /* ofmap padding kernel */
    if (isLast)
      knls.put(cp.name + "_ofmap", addKernel(new PaddingKernel(
          makeKernelParameters(cp.name + "_ofmap"),
          cp.getOfmapStreamBitWidth(),
          cp.dbg)));
  }

  /**
   * Get LayerParameters of LeNet.
   * 
   * @param ep
   * @return
   */
  public static List<LayerParameters> getLayerParameters(
      LeNetEngineParameters ep) {
    List<LayerParameters> lp = new ArrayList<LayerParameters>();

    List<Integer> PP = ep.getPP();

    CompSeq seq0 =
        (ep.getUseDepth()) ? CompSeq.FILTER_MAJOR : CompSeq.CHANNEL_MAJOR;

    ConvLayerParameters cp0 =
        new ConvLayerParameters.Builder(28, 28, 1, 32, 5)
            .name("conv0")
            .BW(ep.getBW())
            .PK(2)
            .PF(PP.get(0))
            .pool(new PoolingLayerParameters(2, 2, Mode.MAX))
            .seq(seq0)
            .type(ep.getUseDepth() ? Type.DEPTHWISE_SEPARABLE : Type.STANDARD)
            .dbg(false)
            .build();
    lp.add(cp0);

    ConvLayerParameters cp1 =
        new ConvLayerParameters.Builder(12, 12, 32, 64, 5)
            .name("conv1")
            .BW(ep.getBW())
            .PK(1)
            .PC(PP.get(0))
            .PF(1)
            .seq(CompSeq.FILTER_MAJOR)
            .pool(new PoolingLayerParameters(2, 2, Mode.MAX))
            .type(ep.getUseDepth() ? Type.DEPTHWISE_SEPARABLE : Type.STANDARD)
            .dbg(false)
            .build();
    lp.add(cp1);

    FullyConnectedLayerParameters fp0 =
        new FullyConnectedLayerParameters(
            "fp0",
            ep.getBW(),
            1024,
            4 * 4 * 64,
            1,
            PP.get(1));
    lp.add(fp0);

    FullyConnectedLayerParameters fp1 =
        new FullyConnectedLayerParameters(
            "fp1",
            ep.getBW(),
            10,
            1024,
            PP.get(1),
            PP.get(2));
    lp.add(fp1);

    return lp;
  }

  private Map<String, LMemCommandGroup> createLMemGroups(
      List<LayerParameters> lps) {
    Map<String, LMemCommandGroup> groups =
        new HashMap<String, LMemCommandGroup>();

    /* LMem connections */
    LMemInterface iface = addLMemInterface();
    /* ifmap group */
    groups.put("ifmap", iface.addCommandGroup(
        "ifmap",
        LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
    /* ofmap group */
    groups.put("ofmap", iface.addCommandGroup(
        "ofmap",
        LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));

    createConvLayerLMemCommandGroup(
        groups,
        (ConvLayerParameters) lps.get(0),
        iface);
    createConvLayerLMemCommandGroup(
        groups,
        (ConvLayerParameters) lps.get(1),
        iface);
    createFullyConnectedLayerLMemCommandGroup(
        groups,
        (FullyConnectedLayerParameters) lps.get(2),
        iface);
    createFullyConnectedLayerLMemCommandGroup(
        groups,
        (FullyConnectedLayerParameters) lps.get(3),
        iface);

    return groups;
  }

  private void createFullyConnectedLayerLMemCommandGroup(
      Map<String, LMemCommandGroup> grps,
      FullyConnectedLayerParameters fp,
      LMemInterface iface) {
    grps.put(fp.name + "_coeff", iface.addCommandGroup(
        fp.name + "_coeff",
        LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
  }

  private void createConvLayerLMemCommandGroup(
      Map<String, LMemCommandGroup> grps,
      ConvLayerParameters cp,
      LMemInterface iface) {
    if (cp.type == Type.STANDARD)
      grps.put(cp.name + "_coeff", iface.addCommandGroup(
          cp.name + "_coeff",
          LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
    else if (cp.type == Type.DEPTHWISE_SEPARABLE) {
      grps
          .put(cp.name + "_depthwise_coeff", iface.addCommandGroup(
              cp.name + "_depthwise_coeff",
              LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
      grps
          .put(cp.name + "_pointwise_coeff", iface.addCommandGroup(
              cp.name + "_pointwise_coeff",
              LMemCommandGroup.MemoryAccessPattern.LINEAR_1D));
    }

  }

  public EngineInterface interfaceDefault(List<LayerParameters> lps) {
    EngineInterface ei = new EngineInterface();

    InterfaceParam batchSize = ei.addParam("batch_size", CPUTypes.UINT64);

    logMsg("Setting up LMem address map ...");
    Map<String, LMemStreamAddress> addrMap =
        createLMemAddressMap(lps, ei, batchSize);

    logMsg("Setting up kernel ticks ...");
    setKernelTicks(lps, addrMap, ei, batchSize);

    logMsg("Setting up kernel scalars ...");
    setKernelScalars(lps, addrMap, ei, batchSize);

    return ei;
  }

  private void setKernelScalars(
      List<LayerParameters> lps,
      Map<String, LMemStreamAddress> addrMap,
      EngineInterface ei,
      InterfaceParam batchSize) {

    ConvLayerParameters cp0 = (ConvLayerParameters) lps.get(0);
    ConvLayerParameters cp1 = (ConvLayerParameters) lps.get(1);
    FullyConnectedLayerParameters fp0 =
        (FullyConnectedLayerParameters) lps.get(2);
    FullyConnectedLayerParameters fp1 =
        (FullyConnectedLayerParameters) lps.get(3);

    ei.setScalar(
        cp0.name + "_ifmap",
        UnpaddingKernel.SCALAR_NUM_INP,
        cp0.getNumCycles() * batchSize);
    ei.setScalar(
        cp0.name + "_ifmap",
        UnpaddingKernel.SCALAR_TOTAL_CYCLES,
        addrMap.get("IFMAP").size / cp0.getIfmapVecSize());

    ei.setScalar(
        fp1.name + "_ofmap",
        PaddingKernel.SCALAR_NUM_INP,
        fp1.getOfmapStreamNumElems() / fp1.getOfmapVecSize() * batchSize);
    ei.setScalar(
        fp1.name + "_ofmap",
        PaddingKernel.SCALAR_TOTAL_CYCLES,
        addrMap.get("OFMAP").size / fp1.getOfmapVecSize());

    setConvLayerKernelScalars(cp0, addrMap, ei, batchSize);
    setConvLayerKernelScalars(cp1, addrMap, ei, batchSize);
    setFullyConnectedLayerKernelScalars(fp0, addrMap, ei, batchSize);
    setFullyConnectedLayerKernelScalars(fp1, addrMap, ei, batchSize);
  }

  public static void setFullyConnectedLayerKernelScalars(
      FullyConnectedLayerParameters fp,
      Map<String, LMemStreamAddress> addrMap,
      EngineInterface ei,
      InterfaceParam batchSize) {
    ei.setScalar(
        fp.name + "_coeff",
        UnpaddingKernel.SCALAR_NUM_INP,
        fp.getNumCycles() * batchSize);
    ei.setScalar(
        fp.name + "_coeff",
        UnpaddingKernel.SCALAR_TOTAL_CYCLES,
        addrMap.get(fp.name + "_COEFF").size / fp.getCoeffVecSize());

  }

  public static void setConvLayerKernelScalars(
      ConvLayerParameters cp,
      Map<String, LMemStreamAddress> addrMap,
      EngineInterface ei,
      InterfaceParam batchSize) {

    if (cp.type == Type.STANDARD) {
      ei.setScalar(
          cp.name + "_coeff",
          UnpaddingKernel.SCALAR_NUM_INP,
          cp.getNumCycles() * cp.K * cp.K * batchSize);
      ei.setScalar(
          cp.name + "_coeff",
          UnpaddingKernel.SCALAR_TOTAL_CYCLES,
          addrMap.get(cp.name + "_COEFF").size / cp.getCoeffLMemVecSize());
    } else {
      ei.setScalar(
          cp.name + "_depthwise_coeff",
          UnpaddingKernel.SCALAR_NUM_INP,
          cp.getNumCycles() * cp.K * cp.K * batchSize);
      ei.setScalar(
          cp.name + "_depthwise_coeff",
          UnpaddingKernel.SCALAR_TOTAL_CYCLES,
          addrMap.get(cp.name + "_DEPTHWISE_COEFF").size
              / cp.getDepthwiseCoeffLMemVecSize());
      ei.setScalar(
          cp.name + "_pointwise_coeff",
          UnpaddingKernel.SCALAR_NUM_INP,
          cp.getNumCycles() * batchSize);
      ei.setScalar(
          cp.name + "_pointwise_coeff",
          UnpaddingKernel.SCALAR_TOTAL_CYCLES,
          addrMap.get(cp.name + "_POINTWISE_COEFF").size
              / cp.getPointwiseCoeffVecSize());
    }

  }

  private void setKernelTicks(
      List<LayerParameters> lps,
      Map<String, LMemStreamAddress> addrMap,
      EngineInterface ei,
      InterfaceParam batchSize) {

    /* conv0 */
    setConvLayerKernelTicks(
        (ConvLayerParameters) lps.get(0),
        addrMap,
        ei,
        batchSize,
        true,
        false);
    /* conv1 */
    setConvLayerKernelTicks(
        (ConvLayerParameters) lps.get(1),
        addrMap,
        ei,
        batchSize,
        false,
        false);
    /* fp0 */
    setFullyConnectedLayerKernelTicks(
        (FullyConnectedLayerParameters) lps.get(2),
        addrMap,
        ei,
        batchSize,
        false,
        false);
    /* fp1 */
    setFullyConnectedLayerKernelTicks(
        (FullyConnectedLayerParameters) lps.get(3),
        addrMap,
        ei,
        batchSize,
        false,
        true);
  }

  public static void setConvLayerKernelTicks(
      ConvLayerParameters cp,
      Map<String, LMemStreamAddress> addrMap,
      EngineInterface ei,
      InterfaceParam batchSize,
      boolean isFirst,
      boolean isLast) {
    /* set core kernel num ticks */
    ei.setTicks(cp.name, cp.getNumCycles() * batchSize);
    /* if there is a pooling layer, set its ticks */
    if (cp.pool != null)
      ei.setTicks(cp.name + "_pool", cp.getPoolNumCycles() * batchSize);
    /* set coeff unpad */
    if (cp.type == Type.STANDARD) {
      ei.setTicks(
          cp.name + "_coeff",
          addrMap.get(cp.name + "_COEFF").size / cp.getCoeffLMemVecSize());
    } else {
      ei.setTicks(
          cp.name + "_depthwise_coeff",
          addrMap.get(cp.name + "_DEPTHWISE_COEFF").size
              / cp.getDepthwiseCoeffLMemVecSize());
      ei.setTicks(
          cp.name + "_pointwise_coeff",
          addrMap.get(cp.name + "_POINTWISE_COEFF").size
              / cp.getPointwiseCoeffVecSize());
    }
    /* ifmap */
    if (isFirst)
      ei.setTicks(
          cp.name + "_ifmap",
          addrMap.get("IFMAP").size / cp.getIfmapVecSize());
    if (isLast)
      ei.setTicks(
          cp.name + "_ofmap",
          addrMap.get("OFMAP").size / cp.getOfmapVecSize());
  }

  public static void setFullyConnectedLayerKernelTicks(
      FullyConnectedLayerParameters fp,
      Map<String, LMemStreamAddress> addrMap,
      EngineInterface ei,
      InterfaceParam batchSize,
      boolean isFirst,
      boolean isLast) {
    /* core kernel */
    InterfaceParam loopLength =
        ei.getAutoLoopOffset(fp.name, fp.name + "_LOOP_LATENCY");
    ei.ignoreAutoLoopOffset(fp.name, fp.name + "_LOOP_LATENCY");
    ei.setTicks(fp.name, fp.getNumCycles() * batchSize * loopLength);
    /* coeff */
    ei.setTicks(
        fp.name + "_coeff",
        addrMap.get(fp.name + "_COEFF").size / fp.getCoeffVecSize());
    /* ifmap */
    if (isFirst)
      ei.setTicks(
          fp.name + "_ifmap",
          addrMap.get("IFMAP").size / fp.getIfmapVecSize());
    /* ofmap */
    if (isLast)
      ei.setTicks(
          fp.name + "_ofmap",
          addrMap.get("OFMAP").size / fp.getOfmapVecSize());

  }

  public static class LMemStreamAddress {
    public final InterfaceParam base;
    public final InterfaceParam size;

    public LMemStreamAddress(InterfaceParam base, InterfaceParam size) {
      this.base = base;
      this.size = size;
    }
  }

  private Map<String, LMemStreamAddress> createLMemAddressMap(
      List<LayerParameters> lps,
      EngineInterface ei,
      InterfaceParam batchSize) {
    Map<String, LMemStreamAddress> addrMap =
        new HashMap<String, LMemStreamAddress>();

    LMemStreamAddress addr;

    InterfaceParam baseAddr = ei.addConstant(0);

    ConvLayerParameters cp0 = (ConvLayerParameters) lps.get(0);
    ConvLayerParameters cp1 = (ConvLayerParameters) lps.get(1);
    FullyConnectedLayerParameters fp0 =
        (FullyConnectedLayerParameters) lps.get(2);
    FullyConnectedLayerParameters fp1 =
        (FullyConnectedLayerParameters) lps.get(3);

    /* ifmap */
    InterfaceParam ifmapSize =
        ConvLayerManagerUtils.getBurstAlignedNumElems(
            ei.addConstant(cp0.getIfmapStreamNumElems()).cast(CPUTypes.INT64)
                * batchSize,
            cp0.getCPUTypes().sizeInBytes(),
            ei.addConstant(cp0.getIfmapVecSize()).cast(CPUTypes.INT64),
            ei)
            * cp0.getCPUTypes().sizeInBytes();
    addr = new LMemStreamAddress(baseAddr, ifmapSize);
    baseAddr += addr.size;
    addrMap.put("IFMAP", addr);
    ei.setLMemLinear("LMEM_IFMAP", addr.base, addr.size);

    /* ofmap */
    InterfaceParam ofmapSize =
        ConvLayerManagerUtils.getBurstAlignedNumElems(
            ei.addConstant(fp1.getOfmapStreamNumElems()).cast(CPUTypes.INT64)
                * batchSize,
            fp1.getCPUTypes().sizeInBytes(),
            ei.addConstant(fp1.getOfmapVecSize()).cast(CPUTypes.INT64),
            ei)
            * fp1.getCPUTypes().sizeInBytes();
    addr = new LMemStreamAddress(baseAddr, ofmapSize);
    baseAddr += addr.size;
    addrMap.put("OFMAP", addr);
    ei.setLMemLinear("LMEM_OFMAP", addr.base, addr.size);

    /* coeff */
    if (cp0.type == Type.STANDARD) {
      addr = createCoeffLMemAddressMap(cp0, ei, baseAddr, batchSize);
      addrMap.put(cp0.name + "_COEFF", addr);
      baseAddr += addr.size;
      ei.setLMemLinear(cp0.name + "_LMEM_COEFF", addr.base, addr.size);
    } else {
      addr = createCoeffLMemAddressMap(cp0, ei, baseAddr, batchSize, true);
      addrMap.put(cp0.name + "_DEPTHWISE_COEFF", addr);
      baseAddr += addr.size;
      ei
          .setLMemLinear(
              cp0.name + "_LMEM_DEPTHWISE_COEFF",
              addr.base,
              addr.size);

      addr = createCoeffLMemAddressMap(cp0, ei, baseAddr, batchSize, false);
      addrMap.put(cp0.name + "_POINTWISE_COEFF", addr);
      baseAddr += addr.size;
      ei
          .setLMemLinear(
              cp0.name + "_LMEM_POINTWISE_COEFF",
              addr.base,
              addr.size);
    }

    if (cp1.type == Type.STANDARD) {
      addr = createCoeffLMemAddressMap(cp1, ei, baseAddr, batchSize);
      addrMap.put(cp1.name + "_COEFF", addr);
      baseAddr += addr.size;
      ei.setLMemLinear(cp1.name + "_LMEM_COEFF", addr.base, addr.size);
    } else {
      addr = createCoeffLMemAddressMap(cp1, ei, baseAddr, batchSize, true);
      addrMap.put(cp1.name + "_DEPTHWISE_COEFF", addr);
      baseAddr += addr.size;
      ei
          .setLMemLinear(
              cp1.name + "_LMEM_DEPTHWISE_COEFF",
              addr.base,
              addr.size);

      addr = createCoeffLMemAddressMap(cp1, ei, baseAddr, batchSize, false);
      addrMap.put(cp1.name + "_POINTWISE_COEFF", addr);
      baseAddr += addr.size;
      ei
          .setLMemLinear(
              cp1.name + "_LMEM_POINTWISE_COEFF",
              addr.base,
              addr.size);
    }

    addr = createCoeffLMemAddressMap(fp0, ei, baseAddr, batchSize);
    addrMap.put(fp0.name + "_COEFF", addr);
    baseAddr += addr.size;
    ei.setLMemLinear(fp0.name + "_LMEM_COEFF", addr.base, addr.size);

    addr = createCoeffLMemAddressMap(fp1, ei, baseAddr, batchSize);
    addrMap.put(fp1.name + "_COEFF", addr);
    baseAddr += addr.size;
    ei.setLMemLinear(fp1.name + "_LMEM_COEFF", addr.base, addr.size);

    return addrMap;
  }

  public static LMemStreamAddress createCoeffLMemAddressMap(
      LayerParameters lp,
      EngineInterface ei,
      InterfaceParam base,
      InterfaceParam batchSize) {
    InterfaceParam size =
        ConvLayerManagerUtils.getBurstAlignedNumElems(
            ei.addConstant(lp.getCoeffStreamNumElems()).cast(CPUTypes.INT64)
                * batchSize,
            lp.getCPUTypes().sizeInBytes(),
            ei.addConstant(lp.getCoeffVecSize()).cast(CPUTypes.INT64),
            ei)
            * lp.getCPUTypes().sizeInBytes();

    return new LMemStreamAddress(base, size);
  }

  public static LMemStreamAddress createCoeffLMemAddressMap(
      ConvLayerParameters cp,
      EngineInterface ei,
      InterfaceParam base,
      InterfaceParam batchSize) {
    InterfaceParam size =
        ConvLayerManagerUtils.getBurstAlignedNumElems(
            ei.addConstant(cp.getCoeffStreamNumElems()).cast(CPUTypes.INT64)
                * batchSize,
            cp.getCPUTypes().sizeInBytes(),
            ei.addConstant(cp.getCoeffLMemVecSize()).cast(CPUTypes.INT64),
            ei)
            * cp.getCPUTypes().sizeInBytes();

    return new LMemStreamAddress(base, size);
  }

  public static LMemStreamAddress createCoeffLMemAddressMap(
      LayerParameters lp,
      EngineInterface ei,
      InterfaceParam base,
      InterfaceParam batchSize,
      boolean isDepthwise) {

    ConvLayerParameters cp = (ConvLayerParameters) lp;

    long streamNumElems =
        (isDepthwise) ? cp.getDepthwiseCoeffStreamNumElems() : cp
            .getPointwiseCoeffStreamNumElems();
    int vecSize =
        (isDepthwise) ? cp.getDepthwiseCoeffLMemVecSize() : cp
            .getPointwiseCoeffVecSize();

    InterfaceParam size =
        ConvLayerManagerUtils.getBurstAlignedNumElems(
            ei.addConstant(streamNumElems).cast(CPUTypes.INT64) * batchSize,
            lp.getCPUTypes().sizeInBytes(),
            ei.addConstant(vecSize).cast(CPUTypes.INT64),
            ei)
            * lp.getCPUTypes().sizeInBytes();

    return new LMemStreamAddress(base, size);
  }

  @SuppressWarnings("deprecation")
  public static void main(String[] args) {
    System.out.println("Start building LeNet ...");

    LeNetEngineParameters ep = new LeNetEngineParameters(args);
    System.out.println("Parsed arguments");

    /* create layer parameters of LeNet */
    List<LayerParameters> lps = getLayerParameters(ep);

    System.out.println("Start building Manager ...");
    LeNetManager mgr = new LeNetManager(ep, lps);

    System.out.println("Start creating manager interface ...");
    mgr.createSLiCinterface(mgr.interfaceDefault(lps));

    BuildConfig buildConfig = mgr.getBuildConfig();
    buildConfig.setBuildEffort(Effort.HIGH);
    buildConfig.setMPPRCostTableSearchRange(1, 4);
    buildConfig.setMPPRParallelism(4);
    mgr.build();
  }
}
