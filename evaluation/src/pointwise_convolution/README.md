# Pointwise Convolution Evaluation

This is a short summary of this evaluation.

We implement an example that can run tiled pointwise convolution on DFE
with result validation from CPU.

Data type in this example is single-precision floating-point,
but ideally it can be replaced with other types without huge revision.

## Configuration parameters

| Parameter        | Description                           | Note |
|------------------|---------------------------------------|------|
| `TILE_HEIGHT`    | height of the tile                    |      |
| `TILE_WIDTH`     | width of the tile                     |      |
| `TILE_IN_DEPTH`  | input depth of the tile               |      |
| `TILE_OUT_DEPTH` | output depth of the tile              |      |
| `PAR_WIDTH`      | parallelised units along width        |      |
| `PAR_IN_DEPTH`   | parallelised units along input depth  |      |
| `PAR_OUT_DEPTH`  | parallelised units along output depth |      |

## Usage

Run simulation (go to the corresponding build directory `build/pointwise_convolution`:

```shell
make runsim \
  MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/<your username>/builds \
  TILE_HEIGHT=16 \
  TILE_WIDTH=16 \
  TILE_IN_DEPTH=4 \
  TILE_OUT_DEPTH=4 \
  PAR_WIDTH=2 \
  PAR_IN_DEPTH=2 \
  PAR_OUT_DEPTH=2
```

This simulation will run a pointwise convolution by a `16 x 16 x 4 x 4` tile,
in which the computation will be parallelised by `2 x 2 x 2`.
