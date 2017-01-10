# maxdeep

Yet another (or a brand new) MaxJ based deep learning implementation.

The workflow of maxdeep is like this:

1. **Parse configuration**: Take input of a neural network architecture (like `.prototxt` in Caffe);
2. **Optimise architecture**: Decide the optimal hardware structure for the given neural network architecture;
3. **Build hardware**: This is delegated to the MaxCompiler
