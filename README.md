# MaxDeep: A Deep Learning Framework on Maxeler Platform

Please cite the following paper(s) if you use this framework, cheers!

```bibtex
@inproceedings{arc17rz,
  author    = {Ruizhe Zhao and Xinyu Niu and Yajie Wu and Wayne Luk and Qiang Liu},
  title     = {Optimizing CNN-based Object Detection Algorithms on Embedded FPGA Platforms},
  booktitle = {{ARC}},
  year      = {2017}
}
@inproceedings{zhao17deeppump,
  author    = {Ruizhe Zhao and Tim Todman and Wayne Luk and Xinyu Niu},
  title     = {{DeepPump}: Multi-Pumping Deep Neural Networks},
  booktitle = {{ASAP}},
  year      = {2017}
}
```

## Install

### Dependencies

* MaxCompiler: `>= 2016.1.1` (we are using `KernelBase`)

### Work with MaxIDE

1. Import this project folder into your workspace
2. Add `MaxCompiler.jar` (under `/opt/maxeler/lib` usually) and `JUnit4` to your project's library dependencies
3. Check whether it works by running JUnit tests

### Work without MaxIDE

We provide several evaluation sub-projects under `evaluation/`.

## Experiment Results

Please see the [evaluation](evaluation) folder.
