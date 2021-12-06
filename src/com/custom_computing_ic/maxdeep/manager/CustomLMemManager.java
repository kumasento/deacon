package com.custom_computing_ic.maxdeep.manager;

import com.custom_computing_ic.maxdeep.manager.conv_single_layer.ConvSingleLayerEngineParameters;
import com.custom_computing_ic.maxdeep.manager.ManagerInterface;
import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemInterface;

/** @author Ruizhe Zhao */
public
class CustomLMemManager extends CustomManager implements ManagerInterface {
 public
  LMemInterface iface;

 public
  CustomLMemManager(ConvLayerEngineParameters params) { super(params); }
 public
  LMemInterface getLMemInterface() { return iface; }
}
