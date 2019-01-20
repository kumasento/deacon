---
Author: Ruizhe Zhao <vincentzhaorz@gmail.com>
Date: 2019/01/20

---

# Dot-product evaluation

This is an example project of performing evaluation in MaxDeep.

- [Configuration parameters](#configuration-parameters)
- [Simulation](#simulation)
- [Hardware](#hardware)
- [Switching between Max4 and Max5](#switching-between-max4-and-max5)

## Configuration parameters

- `BIT_WIDTH` - bit-width of each element
- `VEC_SIZE` - size of the vector to be computed per-cycle

## Simulation

```shell
make runsim BIT_WIDTH=32 VEC_SIZE=32
```

## Hardware

```shell
make build BIT_WIDTH=32 VEC_SIZE=32 # on CAD
make run BIT_WIDTH=32 VEC_SIZE=32   # on hardware system
```

## Switching between Max4 and Max5

Which Maxeler board should be the building target is configured by `DEVICE` in the `Makefile` under `build/dotprod`.

```shell
make runsim BIT_WIDTH=32 VEC_SIZE=32 DEVICE=Lima
```

When using `Lima` as `DEVICE`, this `Makefile` will automatically configure the parameters for building and running on `MAX5`.