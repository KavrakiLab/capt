# Collision-Affording Point Trees: SIMD-Amenable Nearest Neighbors for Fast Collision Checking

[![arXiv CAPT](https://img.shields.io/badge/arXiv-2406.02807-b31b1b.svg)](https://arxiv.org/abs/2406.02807)

![Demo video](doc/capt_demo.gif)

_For a full demonstration of CAPTs running in real-time, see [this video](https://www.youtube.com/watch?v=BzDKdrU1VpM)_.

This is a Rust implementation of the _collision-affording point tree_ (CAPT), a data structure for
SIMD-parallel collision-checking against point clouds.
The CAPT supports extremely high-throughput collision checking, supporting online, real-time motion planning.

You may also want to look at the following other sources:

- [The paper](https://arxiv.org/abs/2406.02807)
- [Demo video](https://www.youtube.com/watch?v=BzDKdrU1VpM)
- [C++ implementation](https://github.com/KavrakiLab/vamp)
- [Blog post about it](https://www.claytonwramsey.com/blog/captree)

If you use this in an academic work, please cite it as follows:

```bibtex
@InProceedings{capt,
  title = {Collision-Affording Point Trees: {SIMD}-Amenable Nearest Neighbors for Fast Collision Checking},
  author = {Ramsey, Clayton W. and Kingston, Zachary and Thomason, Wil and Kavraki, Lydia E.},
  booktitle = {Robotics: Science and Systems},
  date = {2024},
  url = {https://www.roboticsproceedings.org/rss20/p038.pdf},
}
```

## Usage

The core data structure in this library is the `Capt`, which is a search tree used for collision checking.

```rust
use capt::Capt;

// list of points in tree
let points = [[1.0, 1.0], [2.0, 1.0], [3.0, -1.0]];

// range of legal radii for collision-checking
let radius_range = (0.0, 100.0);

let captree = Capt::new(&points, radius_range);

// sphere centered at (1.5, 1.5) with radius 0.01 does not collide
assert!(!captree.collides(&[1.5, 1.5], 0.01));

// sphere centered at (1.5, 1.5) with radius 1.0 does collide
assert!(captree.collides(&[1.5, 1.5], 0.01));
```

## License

This work is licensed to you under the Apache 2.0 license.
For further details, refer to [LICENSE.md](/LICENSE.md).
