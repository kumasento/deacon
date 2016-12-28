#include <iostream>
#include <MaxSLiCInterface.h>
#include "MaxDeep.h"

#include "maxdeep.pb.h"
#include "glog/logging.h"

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  LOG(INFO) << "Running MaxDeep Simulation ...";

  MaxDeep_actions_t actions;

  return 0;
}
