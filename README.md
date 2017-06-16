# MaxDeep: A Deep Learning Framework on Maxeler Platform

Please cite the following paper(s) if you use this framework, cheers!

```bibtex
@inproceedings{arc17rz,
  author    = {Ruizhe Zhao and Xinyu Niu and Yajie Wu and Wayne Luk and Qiang Liu},
  title     = {Optimizing CNN-based Object Detection Algorithms on Embedded FPGA Platforms},
  booktitle = {{ARC}},
  year      = {2017}
}
```

## Migration Roadmap

Currently I am migrating maxdeep from v2, which is the version I used for
my MRes second project and the ASAP '17 submission, to a new stable version.
The migration process will take several days.
There are several steps to follow:

- [x] migrate and test `LineBuffer`.
- [ ] migrate and test `InputFMapBuffer`.

## Designs Under Exploration (Deprecated)

Currently there are several designs are being developed and experimented with. Each design has its unique name, a parameter list, and a corresponding software-side test function. Each design is aiming at testing some parts of our proposed architecture to get an understanding of how these parts work and how much resource will they consume, etc.

A full list of designs are as follows:

| Name              | Parameters  | Description                                                          |
|-------------------|-------------|----------------------------------------------------------------------|
| `LOOPBACK`        |             | How to work with 3 different streams with LMem                       |
| `LOOPBACK_PADDED` |             | The streams are padded with padding kernels                          |
| `MULT_ARRAY`      | `P`         | Depreacted                                                           |
| `ONE_DIM_CONV`    | `W` `P` `M` | Test parallel, bandwidth, and multi-pumping for one dimensional CONV |
