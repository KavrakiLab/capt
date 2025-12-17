#![feature(portable_simd)]

use std::{
    env,
    error::Error,
    path::Path,
    simd::{LaneCount, Simd, SupportedLaneCount},
    time::{Duration, Instant},
};

use capt::Axis;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use rand_distr::{Distribution, Normal};

pub mod forest;
pub mod kdt;

pub fn get_points(n_points_if_no_cloud: usize) -> Box<[[f32; 3]]> {
    let args: Vec<String> = env::args().collect();
    let mut rng = ChaCha20Rng::seed_from_u64(2707);

    if args.len() > 1 {
        eprintln!("Loading pointcloud from {}", &args[1]);
        parse_pointcloud_csv(&args[1]).unwrap()
    } else {
        eprintln!("No pointcloud file! Using N={n_points_if_no_cloud}");
        eprintln!("generating random points...");
        (0..n_points_if_no_cloud)
            .map(|_| {
                [
                    rng.random_range::<f32, _>(0.0..1.0),
                    rng.random_range::<f32, _>(0.0..1.0),
                    rng.random_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect()
    }
}

/// Generate some randomized numbers for us to benchmark against.
///
/// # Generic parameters
///
/// - `D`: the dimension of the space
/// - `L`: the number of SIMD lanes
///
/// # Returns
///
/// Returns a pair `(seq_needles, simd_needles)`, where `seq_needles` is correctly shaped for
/// sequential querying and `simd_needles` is correctly shaped for SIMD querying.
pub fn make_needles<const D: usize, const L: usize>(
    rng: &mut impl Rng,
    n_trials: usize,
) -> (Vec<[f32; D]>, Vec<[Simd<f32, L>; D]>)
where
    LaneCount<L>: SupportedLaneCount,
{
    let mut seq_needles = Vec::new();
    let mut simd_needles = Vec::new();

    for _ in 0..n_trials / L {
        let mut simd_pts = [Simd::splat(0.0); D];
        for l in 0..L {
            let mut seq_needle = [0.0; D];
            for d in 0..3 {
                let value = rng.random_range::<f32, _>(0.0..1.0);
                seq_needle[d] = value;
                simd_pts[d].as_mut_array()[l] = value;
            }
            seq_needles.push(seq_needle);
        }
        simd_needles.push(simd_pts);
    }

    assert_eq!(seq_needles.len(), simd_needles.len() * L);

    (seq_needles, simd_needles)
}

/// Generate some randomized numbers for us to benchmark against which are correlated within a SIMD
/// batch.
///
/// # Generic parameters
///
/// - `D`: the dimension of the space
/// - `L`: the number of SIMD lanes
///
/// # Returns
///
/// Returns a pair `(seq_needles, simd_needles)`, where `seq_needles` is correctly shaped for
/// sequential querying and `simd_needles` is correctly shaped for SIMD querying.
/// Additionally, each element of each element of `simd_needles` will be relatively close in space.
pub fn make_correlated_needles<const D: usize, const L: usize>(
    rng: &mut impl Rng,
    n_trials: usize,
) -> (Vec<[f32; D]>, Vec<[Simd<f32, L>; D]>)
where
    LaneCount<L>: SupportedLaneCount,
{
    let mut seq_needles = Vec::new();
    let mut simd_needles = Vec::new();

    for _ in 0..n_trials / L {
        let mut start_pt = [0.0; D];
        for v in start_pt.iter_mut() {
            *v = rng.random_range::<f32, _>(0.0..1.0);
        }
        let mut simd_pts = [Simd::splat(0.0); D];
        for l in 0..L {
            let mut seq_needle = [0.0; D];
            for d in 0..D {
                let value = start_pt[d] + rng.random_range::<f32, _>(-0.02..0.02);
                seq_needle[d] = value;
                simd_pts[d].as_mut_array()[l] = value;
            }
            seq_needles.push(seq_needle);
        }
        simd_needles.push(simd_pts);
    }

    assert_eq!(seq_needles.len(), simd_needles.len() * L);

    (seq_needles, simd_needles)
}

pub fn dist<const D: usize>(a: [f32; D], b: [f32; D]) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(x1, x2)| (x1 - x2).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn stopwatch<F: FnOnce() -> R, R>(f: F) -> (R, Duration) {
    let tic = Instant::now();
    let r = f();
    (r, Instant::now().duration_since(tic))
}

pub fn parse_pointcloud_csv(
    p: impl AsRef<Path>,
) -> Result<Box<[[f32; 3]]>, Box<dyn std::error::Error>> {
    std::str::from_utf8(&std::fs::read(&p)?)?
        .lines()
        .map(|l| {
            let mut split = l.split(',').flat_map(|s| s.parse::<f32>().ok());
            Ok::<_, Box<dyn Error>>([
                split.next().ok_or("trace missing x")?,
                split.next().ok_or("trace missing y")?,
                split.next().ok_or("trace missing z")?,
            ])
        })
        .collect()
}

pub type Trace = [([f32; 3], f32)];

pub fn parse_trace_csv(p: impl AsRef<Path>) -> Result<Box<Trace>, Box<dyn std::error::Error>> {
    std::str::from_utf8(&std::fs::read(&p)?)?
        .lines()
        .map(|l| {
            let mut split = l.split(',').flat_map(|s| s.parse::<f32>().ok());
            Ok::<_, Box<dyn Error>>((
                [
                    split.next().ok_or("trace missing x")?,
                    split.next().ok_or("trace missing y")?,
                    split.next().ok_or("trace missing z")?,
                ],
                split.next().ok_or("trace missing r")?,
            ))
        })
        .collect()
}

pub type SimdTrace<const L: usize> = [([Simd<f32, L>; 3], Simd<f32, L>)];

pub fn simd_trace_new<const L: usize>(trace: &Trace) -> Box<SimdTrace<L>>
where
    LaneCount<L>: SupportedLaneCount,
{
    trace
        .chunks(L)
        .map(|w| {
            let mut centers = [[0.0; L]; 3];
            let mut radii = [0.0; L];
            for (l, ([x, y, z], r)) in w.iter().copied().enumerate() {
                centers[0][l] = x;
                centers[1][l] = y;
                centers[2][l] = z;
                radii[l] = r;
            }
            (centers.map(Simd::from_array), Simd::from_array(radii))
        })
        .collect()
}

pub fn trace_r_range(t: &Trace) -> (f32, f32) {
    (
        t.iter()
            .map(|x| x.1)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0),
        t.iter()
            .map(|x| x.1)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f32::INFINITY),
    )
}

pub fn fuzz_pointcloud(t: &mut [[f32; 3]], stddev: f32, rng: &mut impl Rng) {
    let normal = Normal::new(0.0, stddev).unwrap();
    t.iter_mut()
        .for_each(|p| p.iter_mut().for_each(|x| *x += normal.sample(rng)))
}

#[inline]
/// Calculate the "true" median (halfway between two midpoints) and partition `points` about said
/// median along axis `d`.
fn median_partition<A: Axis, const K: usize>(points: &mut [[A; K]], k: usize) -> A {
    let (lh, med_hi, _) =
        points.select_nth_unstable_by(points.len() / 2, |a, b| a[k].partial_cmp(&b[k]).unwrap());
    let med_lo = lh
        .iter_mut()
        .map(|p| p[k])
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    A::in_between(med_lo, med_hi[k])
}

#[inline]
fn forward_pass<A: Axis, const K: usize>(tests: &[A], point: &[A; K]) -> usize {
    // forward pass through the tree
    let mut test_idx = 0;
    let mut k = 0;
    for _ in 0..tests.len().trailing_ones() {
        test_idx =
            2 * test_idx + 1 + usize::from(unsafe { *tests.get_unchecked(test_idx) } <= point[k]);
        k = (k + 1) % K;
    }

    // retrieve affordance buffer location
    test_idx - tests.len()
}

fn distsq<const K: usize>(a: [f32; K], b: [f32; K]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..K {
        total += (a[i] - b[i]).powi(2);
    }
    total
}
