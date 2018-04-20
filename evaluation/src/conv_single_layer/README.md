# Single Convolution Layer Evaluation

This package evaluates the correctness of a single convolution layer.

The objective is, we test the correctness of a single convolution layer
under different configurations.
Performance and resource usage numbers will be listed.

We define a convolution layer accelerator by

```
ConvLayer(TH, TW, TC, TF, K, S, PW, PC, PF, TYPE, DRAM, WINO)
```

Here `TH` and `TW` are height and width of the output tile.

For each configuration,
we evaluate the following cases:

1. single tile: no tiling is required.
2. multiple tiles: mainly tests the tiling implementation.

## Results

| Configuration                                | Perf (Single) | Perf (Multi) | LUT  | FF    | BRAM | DSP |
|----------------------------------------------|---------------|--------------|------|-------|------|-----|
| 32, 32, 32, 32, 3, 1, 1, 1, 1, `INT16`, N, N | 0.4785        |              | 9605 | 17550 | 157  | 5   |

## Trivia

1. In the configuration, we don't set padding since we 
_manually_ pad in the tiling function.
Therefore, if `TH = TW = 32`, the input tile size should
be `34 x 34` no matter what the padding size is.
