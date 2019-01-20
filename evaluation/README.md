# MaxDeep Evaluation

MaxDeep is evaluated by several seperated projects under this folder.
Each project is wrapped in a `package` and listed under `evaluation/src`.
In each evaluation project package, there will be:

1. A customised `Kernel` that wraps components to be tested
2. A `Manager` that manages the kernel
3. A software code snippet to call the functionality to be evaluated.

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Example projects](#example-projects)
  - [Dot-product](#dot-product)
- [Current Evaluations](#current-evaluations)

## Installation

### Prerequisites

- `GCC >= 4.8.5`
- `GLOG`

## Example projects

You can get an idea about what an evaluation project looks like by checking the following example projects.

### Dot-product

This project implements a basic dot-product processor on FPGA [[README](./src/dotprod)].

## Current Evaluations

- [Single convolution layer](src/conv_single_layer/README.md):
evaluate the correctness of a single convolution layer.

