#include <iostream>
// #include <MaxSLiCInterface.h>
// #include "Maxfiles.h"

#include "maxdeep.pb.h"
#include "glog/logging.h"

int main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  LOG(INFO) << "Running MaxDeep Simulation ...";

  return 0;
}
