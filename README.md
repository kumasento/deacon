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

## Development environment

Instead of jumping into how to build and install Deacon, I would like to have an overview of what the development environment that Deacon would expect. To me (and hopefully you) the development environment is more crucial: it can limit your imagination on what you can do with Deacon!

(P.S. I need this section badly to give myself a reminder :smile:)

### Java

First thing first, you should have Java installed, including JDK and JRE. The version should be at least 1.8.

### MaxCompiler

Next, obviously, you need MaxCompiler. The version I'm using is 2020.2, although not all source code have been upgraded to that version yet, but we will be there. Remember to source the settings of MaxCompiler in the bash before you start any run:

```sh
source /vol/cc/opt/maxeler/maxcompiler-2020.2/settings.sh
```

Yes, my MaxCompiler is installed at `/vol/cc/opt/maxeler/maxcompiler-2020.2`. Make sure you replace that to where yours is installed (hint: it could be under `/opt`).

Another important thing: replace the paths to MaxCompiler's JARs in [.classpath](.classpath). Specifically, change `/vol/cc/opt/maxeler/maxcompiler-2020.2` to where your MaxCompiler is installed. You can ignore this step though, if you don't want to enjoy the smart features provided by IDE/editors: they need `.classpath` to figure out how to resolve your source files.

```xml
<classpathentry kind="lib" path="/vol/cc/opt/maxeler/maxcompiler-2020.2/lib/MaxCompiler.jar" />
<classpathentry kind="lib" path="/vol/cc/opt/maxeler/maxcompiler-2020.2/lib/Max4Platform.jar" />
<classpathentry kind="lib" path="/vol/cc/opt/maxeler/maxcompiler-2020.2/lib/Max5Platform.jar" />
```

### Editor/IDE

A war can start under this title, trust me.

Put jokes aside, the editor I'm using is VSCode. Deacon is placed on a remote server and I find VSCode pretty handy to provide a stable remote editing environment. It also have a rather simple, but enough to use Java support extensions ([list](https://code.visualstudio.com/docs/java/extensions)). At least you should have:

* Basic language server: https://marketplace.visualstudio.com/items?itemName=redhat.java
* Project manager: https://marketplace.visualstudio.com/items?itemName=vscjava.vscode-java-dependency

If you set the `.classpath` right, these extensions can do their work as expected, e.g., you could open [DotProductKernel.java](app/dot_product/DotProductKernel.java) and hover the pointer to any type (`Kernel`, for instance), and there will be a popup window shows where the class `Kernel` is defined.
