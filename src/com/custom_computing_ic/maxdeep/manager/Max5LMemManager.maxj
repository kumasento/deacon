package com.custom_computing_ic.maxdeep.manager;

import com.custom_computing_ic.maxdeep.manager.ManagerInterface;

import com.maxeler.maxcompiler.v2.build.EngineParameters;
import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemInterface;
import com.maxeler.platform.max5.manager.Max5LimaManager;

public
class Max5LMemManager extends Max5LimaManager implements ManagerInterface {
 public
  LMemInterface iface;

 public
  Max5LMemManager(EngineParameters params) { super(params); }

 public
  LMemInterface getLMemInterface() {
    if (iface == null) iface = addLMemInterface();
    return iface;
  }
}