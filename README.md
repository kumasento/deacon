# Deacon: A Deep Learning Library for MaxCompiler

[![Build and Test](https://github.com/kumasento/deacon/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/kumasento/maxdeep/actions/workflows/buildAndTest.yml)

> We recently changed the name of the project from _maxdeep_ to _deacon_. For explanation of the naming, see [here](#naming).

Deacon is a DL library. It has building blocks for constructing DNN with [MaxCompiler](https://www.maxeler.com/products/software/maxcompiler/). MaxCompiler enforces a data-flow paradigm for hardware description (using its internal Java-based HDL [MaxJ](https://www.doc.ic.ac.uk/~georgig/OpenSPL2014/)), on which there is no publicly DL library yet.

With Deacon, you can leverage better powerful FPGA accelerators in Maxeler platforms within the DL application domain.

Experimental data for the latest Maxeler hardware (MAX5, internally it has a [Xilinx VU9P](https://www.xilinx.com/products/silicon-devices/fpga/virtex-ultrascale-plus.html) board) will be released in the coming months. Stay tuned!


Please consider citing the following paper(s) if you use this framework, cheers! :heart:

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
@inproceedings{zhao18towards,
  author    = {Zhao, Ruizhe and Ng, Ho-cheung and Luk, Wayne and Niu, Xinyu},
  title     = {{Towards Efficient Convolutional Neural Network for Domain-Specific Applications on FPGA}},
  year      = {2018}
  booktitle = {{FPL}},
}
```

## Naming

We mainly get this name _Deacon_ from a boss enemy [Deacons of the Deep](https://darksouls3.wiki.fextralife.com/Deacons+of+the+Deep) in Dark Souls 3, one of the author's favourite games. The original meaning of deacon is [servant](https://en.wikipedia.org/wiki/Deacon#Origin_and_development), which perfectly matches the objective of this library as a helper for building _deep_ learning applications.
