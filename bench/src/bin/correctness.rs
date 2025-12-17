#![feature(portable_simd)]

use std::simd::Simd;

use bench::{dist, kdt::PkdTree, parse_pointcloud_csv, parse_trace_csv, trace_r_range};
use capt::Capt;
use kiddo::SquaredEuclidean;
use rand::{seq::SliceRandom, Rng, SeedableRng};

const N: usize = 1 << 12;
const R: f32 = 0.02;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(2707);
    let args = std::env::args().collect::<Box<[String]>>();
    let points: Box<[[f32; 3]]> = if args.len() < 2 {
        (0..1 << 16)
            .map(|_| {
                [
                    rng.random_range::<f32, _>(0.0..1.0),
                    rng.random_range::<f32, _>(0.0..1.0),
                    rng.random_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect()
    } else {
        let mut p = parse_pointcloud_csv(&args[1])?.to_vec();
        p.shuffle(&mut rng);
        p.truncate(1 << 16);
        p.into_boxed_slice()
    };

    let trace: Box<[([f32; 3], f32)]> = if args.len() < 3 {
        (0..N)
            .map(|_| {
                (
                    [
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                    ],
                    rng.random_range(0.0..=R),
                )
            })
            .collect()
    } else {
        parse_trace_csv(&args[2])?
    };

    let r_range = trace_r_range(&trace);

    let kdt = PkdTree::new(&points);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in points.iter() {
        kiddo_kdt.add(pt, 0);
    }

    let aff_tree = Capt::<3>::new(&points, r_range);

    for (i, (center, r)) in trace.iter().enumerate() {
        let exact_kiddo_dist = kiddo_kdt
            .nearest_one::<SquaredEuclidean>(center)
            .distance
            .sqrt();
        let exact_dist = dist(kdt.get_point(kdt.query1_exact(*center)), *center);
        assert_eq!(exact_dist, exact_kiddo_dist);

        let simd_center: [Simd<f32, 8>; 3] = [
            Simd::splat(center[0]),
            Simd::splat(center[1]),
            Simd::splat(center[2]),
        ];
        if exact_dist <= *r {
            println!("iter {i}: {:?} (collides)", (center, r));
            assert!(aff_tree.collides(center, *r));
            assert!(aff_tree.collides_simd(&simd_center, Simd::splat(*r)))
        } else {
            println!("iter {i}: {:?} (no collides)", (center, r));
            assert!(!aff_tree.collides(center, *r));
            assert!(!aff_tree.collides_simd(&simd_center, Simd::splat(*r)))
        }
    }

    Ok(())
}
