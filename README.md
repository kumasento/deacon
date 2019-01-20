# MaxDeep: A Deep Learning Framework on Maxeler Platform

Please consider citing the following paper(s) if you use this framework, cheers! :heart:

```tex
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
@inproceedings{zhao18towards,
  author    = {Zhao, Ruizhe and Ng, Ho-cheung and Luk, Wayne and Niu, Xinyu},
  title     = {{Towards Efficient Convolutional Neural Network for Domain-Specific Applications on FPGA}},
  year      = {2018}
  booktitle = {{FPL}},
}
```

## Install

### Dependencies

* MaxCompiler: `>= 2016.1.1` (we are using `KernelBase`)

### Work with MaxIDE

1. Import this project folder into your workspace (`General > Old/Non MaxCompiler projects`)
2. Add `MaxCompiler.jar` (under `/opt/maxeler/lib` usually, add by `Add External JARs`) and `JUnit4` to your project's library dependencies (add by `Add Library`)
3. Set the following directories as project `Source` (`properties > Java Build Path`):
    1. `maxdeep/src`
    2. `maxdeep/evaluation/src`
    3. `maxdeep/lib/dfe-snippets/src`

### Work without MaxIDE

We provide several evaluation sub-projects under `evaluation/`.

## Experiment Results

Please see the [evaluation](evaluation) folder.
