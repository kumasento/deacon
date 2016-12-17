# maxdeep

Yet another (or a brand new) MaxJ based deep learning implementation.

The workflow of maxdeep is like this:

1. **Parse configuration**: Take input of a neural network architecture (like `.prototxt` in Caffe);
2. **Optimise architecture**: Decide the optimal hardware structure for the given neural network architecture;
3. **Build hardware**: This is delegated to the MaxCompiler

## Route Map

- [x] Finish the basic kernel design for CONV, FC, POOL, and RELU.
- [ ] Complete a basic version of configuration parser, which can only parse a limited number of parameters (type of the layer, input and output connections, feature maps shape, etc.), and assume all layers can be deployed on the given platform (unconstrained optimisation).
- [ ] Construct a LeNet for the MNIST dataset, and a VGGNet. At this stage, just be sure that the network can be compiled on the hardware or not.
- [ ] Check the correctness by extracting data from the forward pass in Caffe.
- [ ] Explore other optimisation techniques, first by using fixed point, then by using LMem.

## Reminders

1. Occam' razor 
