package com.custom_computing_ic.maxdeep.manager;

import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerPCIe;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerSlic;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerRouting;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerBuild;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerLogging;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerLMem;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerKernel;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerSM;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerSimulation;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemGlobalConfig;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemConfig;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemInterface;

/**
 * A uniformed manager interface.
 *
 * Following the advices from Nils
 * https://groups.google.com/a/maxeler.com/forum/#!topic/mdx/sleKzbZFOjA
 */
public
interface ManagerInterface extends ManagerPCIe, ManagerSlic, ManagerRouting,
    ManagerBuild, ManagerLogging, ManagerKernel, ManagerSM, ManagerSimulation {

  // Board specific manager will implement this function.
  LMemInterface addLMemInterface();

  LMemInterface getLMemInterface();
}