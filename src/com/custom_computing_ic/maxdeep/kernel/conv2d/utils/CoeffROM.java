package com.custom_computing_ic.maxdeep.kernel.conv2d.utils;

import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters;
import com.custom_computing_ic.maxdeep.kernel.conv2d.ConvLayerParameters.Type;
import com.custom_computing_ic.maxdeep.kernel.fuse.FusedConvLayerParameters;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelComponent;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.core.CounterChain;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.utils.Bits;
import com.maxeler.maxcompiler.v2.utils.MathUtils;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class CoeffROM {
  /** Rom related. */

  public static double[] readFile(String key, String fileName) {
    Scanner in = null;

    try {
      in = new Scanner(new File(fileName));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    // TODO: properly throw.

    // Find the correct line.
    String line = in.nextLine();
    while (!(line.startsWith("BEGIN") && line.contains(key))) {
      if (!in.hasNext())
        break;

      line = in.nextLine();
    }

    if (!in.hasNext())
      throw new IllegalArgumentException(
          String.format("Cannot find key: %s from file: %s", key, fileName));

    int numElems = in.nextInt();

    double[] rawData = new double[numElems];
    for (int i = 0; i < numElems; ++i) rawData[i] = in.nextDouble();

    return rawData;
  }

  public static boolean isTernaryType(DFEType T) {
    return T.getTotalBits() == 2 && T.isUInt();
  }

  public static boolean isTernaryType(DFEVectorType<DFEVar> T) {
    return isTernaryType((DFEType) T.getContainedType());
  }

  public static double convert(double data, DFEVectorType<DFEVar> T) {
    if (isTernaryType(T))
      return data > 0 ? 1 : (data < 0 ? 3 : 0);
    return data;
  }

  public static Bits[] memData(
      ConvLayerParameters cp, String key, int depth, DFEVectorType<DFEVar> vt) {
    return memData(cp, key, depth, vt, 0, 0);
  }

  public static Bits[] create(
      ConvLayerParameters cp, String key, int depth, DFEVectorType<DFEVar> vt, int ci) {
    return memData(cp, key, depth, vt, ci, 0);
  }

  /**
   * If depth is smaller than 2, we will pad the memory depth to 2 and fill the padded area with
   * invalid data.
   * @param cp
   * @param key
   * @param depth
   * @param vt
   * @param ci
   * @param fi
   * @return
   */
  public static Bits[] memData(
      ConvLayerParameters cp, String key, int depth, DFEVectorType<DFEVar> vt, int ci, int fi) {
    double[] rawData = new double[vt.getSize() * depth];
    double[] srcArray = readFile(key, cp.coeffFile);
    System.arraycopy(srcArray, 0, rawData, 0, Math.min(srcArray.length, rawData.length));

    if (rawData.length % depth != 0)
      throw new IllegalArgumentException("number of data should be divisible by memory depth.");

    double[][] parts = new double[depth][rawData.length / depth];

    // The total number of channels should be scaled.
    int C = cp.C(ci);
    int F = cp.F(0); // F doesn't scale
    int PC = cp.PC.get(ci);
    int PF = cp.PF.get(fi);

    if (cp.type == Type.DEPTHWISE_SEPARABLE) {
      for (int pc = 0; pc < PC; ++pc)
        for (int k = 0; k < cp.K * cp.K; ++k)
          for (int c = 0; c < C; c += PC)
            parts[c / PC][pc * cp.K * cp.K + k] =
                convert(rawData[(c + pc) * (cp.K * cp.K) + k], vt);
    } else {
      for (int pf = 0; pf < PF; ++pf)
        for (int pc = 0; pc < PC; ++pc)
          for (int k = 0; k < cp.K * cp.K; ++k)
            for (int f = 0; f < F; f += PF)
              for (int c = 0; c < C; c += PC) {
                // Here C is padded to calculate the right number of C vectors.
                int row = (f / PF) * (cp.padC(ci) / PC) + (c / PC);
                int col = pf * PC * cp.K * cp.K + pc * cp.K * cp.K + k;
                if (f + pf >= F || c + pc >= C)
                  parts[row][col] = 0;
                else
                  // Note that it is C not padded C being multiplied.
                  parts[row][col] =
                      convert(rawData[(f + pf) * C * cp.K * cp.K + (c + pc) * cp.K * cp.K + k], vt);
              }
    }

    Bits[] memData = new Bits[Math.max(depth, 2)];
    for (int i = 0; i < depth; ++i) memData[i] = vt.encodeConstant(parts[i]);
    for (int i = depth; i < Math.max(depth, 2); ++i) memData[i] = vt.encodeConstant(parts[0]);

    return memData;
  }
}
