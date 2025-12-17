#![feature(portable_simd)]

use std::{
    cmp::min, env::args, error::Error, fs::File, hint::black_box, io::Write, simd::f32x8,
    time::Duration,
};

use bench::{
    forest::PkdForest, fuzz_pointcloud, kdt::PkdTree, parse_pointcloud_csv, parse_trace_csv,
    simd_trace_new, stopwatch, SimdTrace, Trace,
};
use capt::Capt;
#[allow(unused_imports)]
use kiddo::SquaredEuclidean;
use morton_filter::morton_filter;
use rand::{seq::SliceRandom, Rng};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

const N_TRIALS: usize = 100_000;
const L: usize = 8;

const QUERY_RADIUS: f32 = 0.05;

struct Benchmark<'a> {
    seq: &'a Trace,
    simd: &'a SimdTrace<L>,
    f_query: File,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut f_construct = File::create("construct_time.csv")?;
    let mut f_mem = File::create("mem.csv")?;

    let args: Vec<String> = args().collect();

    let mut rng = ChaCha20Rng::seed_from_u64(2707);
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
        fuzz_pointcloud(&mut p, 0.001, &mut rng);
        p.shuffle(&mut rng);
        p.truncate(1 << 16);
        p.into_boxed_slice()
    };

    let all_trace: Box<[([f32; 3], f32)]> = if args.len() < 3 {
        (0..N_TRIALS)
            .map(|_| {
                (
                    [
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                    ],
                    rng.random_range(0.0..=QUERY_RADIUS),
                )
            })
            .collect()
    } else {
        parse_trace_csv(&args[2])?
    };

    // let rsq_range = (
    //     all_trace
    //         .iter()
    //         .map(|x| x.1)
    //         .min_by(|a, b| a.partial_cmp(b).unwrap())
    //         .ok_or("no points")?
    //         .powi(2),
    //     all_trace
    //         .iter()
    //         .map(|x| x.1)
    //         .max_by(|a, b| a.partial_cmp(b).unwrap())
    //         .ok_or("no points")?
    //         .powi(2),
    // );
    let r_range = (0.01, 0.08);

    println!("number of points: {}", points.len());
    println!("number of tests: {}", all_trace.len());
    println!("radius range: {r_range:?}");

    let captree = Capt::<3>::new(&points, r_range);

    let collide_trace: Box<Trace> = all_trace
        .iter()
        .filter(|(center, r)| captree.collides(center, *r))
        .copied()
        .collect();

    let no_collide_trace: Box<Trace> = all_trace
        .iter()
        .filter(|(center, r)| !captree.collides(center, *r))
        .copied()
        .collect();

    let all_simd_trace = simd_trace_new(&all_trace);
    let collide_simd_trace = simd_trace_new(&collide_trace);
    let no_collide_simd_trace = simd_trace_new(&no_collide_trace);

    let mut benchmarks = [
        Benchmark {
            seq: &all_trace,
            simd: &all_simd_trace,
            f_query: File::create("mixed.csv")?,
        },
        Benchmark {
            seq: &collide_trace,
            simd: &collide_simd_trace,
            f_query: File::create("collides.csv")?,
        },
        Benchmark {
            seq: &no_collide_trace,
            simd: &no_collide_simd_trace,
            f_query: File::create("no_collides.csv")?,
        },
    ];

    let mut r_filter = 0.001;
    loop {
        let mut new_points = points.to_vec();
        morton_filter(&mut new_points, r_filter);
        do_row(
            &new_points,
            &mut benchmarks,
            r_range,
            &mut f_construct,
            &mut f_mem,
        )?;
        r_filter *= 1.02;
        if new_points.len() < 500 {
            break;
        }
    }

    Ok(())
}

fn do_row(
    points: &[[f32; 3]],
    benchmarks: &mut [Benchmark],
    r_range: (f32, f32),
    f_construct: &mut File,
    f_mem: &mut File,
) -> Result<(), Box<dyn Error>> {
    let (kdt, kdt_time) = stopwatch(|| kiddo::ImmutableKdTree::new_from_slice(points));

    let (pkdt, pkdt_time) = stopwatch(|| PkdTree::new(points));

    let (captree, captree_time) = stopwatch(|| Capt::<3, L, f32, u32>::new(points, r_range));

    let (f1, f1_time) = stopwatch(|| PkdForest::<3, 1>::new(points));
    let (f2, f2_time) = stopwatch(|| PkdForest::<3, 2>::new(points));
    let (f3, f3_time) = stopwatch(|| PkdForest::<3, 3>::new(points));
    let (f4, f4_time) = stopwatch(|| PkdForest::<3, 4>::new(points));
    let (f5, f5_time) = stopwatch(|| PkdForest::<3, 5>::new(points));
    let (f6, f6_time) = stopwatch(|| PkdForest::<3, 6>::new(points));
    let (f7, f7_time) = stopwatch(|| PkdForest::<3, 7>::new(points));
    let (f8, f8_time) = stopwatch(|| PkdForest::<3, 8>::new(points));
    let (f9, f9_time) = stopwatch(|| PkdForest::<3, 9>::new(points));
    let (f10, f10_time) = stopwatch(|| PkdForest::<3, 10>::new(points));

    writeln!(
        f_construct,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        points.len(),
        kdt_time.as_secs_f64(),
        pkdt_time.as_secs_f64(),
        captree_time.as_secs_f64(),
        f1_time.as_secs_f64(),
        f2_time.as_secs_f64(),
        f3_time.as_secs_f64(),
        f4_time.as_secs_f64(),
        f5_time.as_secs_f64(),
        f6_time.as_secs_f64(),
        f7_time.as_secs_f64(),
        f8_time.as_secs_f64(),
        f9_time.as_secs_f64(),
        f10_time.as_secs_f64(),
    )?;

    writeln!(
        f_mem,
        "{},{},{}",
        points.len(),
        pkdt.memory_used(),
        captree.memory_used()
    )?;

    for Benchmark {
        seq: trace,
        simd: simd_trace,
        f_query,
    } in benchmarks
    {
        let (_, kdt_within_q_time) = stopwatch(|| {
            for (center, radius) in trace.iter() {
                black_box(
                    kdt.within_unsorted::<SquaredEuclidean>(center, radius.powi(2))
                        .is_empty(),
                );
            }
        });
        let (_, kdt_nearest_q_time) = stopwatch(|| {
            for (center, radius) in trace.iter() {
                black_box(kdt.nearest_one::<SquaredEuclidean>(center).distance <= radius.powi(2));
            }
        });
        let kdt_total_q_time = min(kdt_within_q_time, kdt_nearest_q_time);

        let (_, pkdt_total_seq_q_time) = stopwatch(|| {
            for (center, radius) in trace.iter() {
                black_box(pkdt.might_collide(*center, radius.powi(2)));
            }
        });
        let (_, pkdt_total_simd_q_time) = stopwatch(|| {
            for (centers, radii) in simd_trace.iter() {
                black_box(pkdt.might_collide_simd(centers, radii * radii));
            }
        });
        let (_, captree_total_seq_q_time) = stopwatch(|| {
            for (center, radius) in trace.iter() {
                black_box(captree.collides(center, radius.powi(2)));
            }
        });
        let (_, captree_total_simd_q_time) = stopwatch(|| {
            for (centers, radii) in simd_trace.iter() {
                black_box(captree.collides_simd(centers, radii * radii));
            }
        });

        let trace_len = trace.len() as f64;
        write!(
            f_query,
            "{},{},{},{},{},{},{}",
            points.len(),
            trace.len(),
            kdt_total_q_time.as_secs_f64() / trace_len,
            pkdt_total_seq_q_time.as_secs_f64() / trace_len,
            pkdt_total_simd_q_time.as_secs_f64() / trace_len,
            captree_total_seq_q_time.as_secs_f64() / trace_len,
            captree_total_simd_q_time.as_secs_f64() / trace_len,
        )?;

        let forest_results = [
            bench_forest(&f1, simd_trace),
            bench_forest(&f2, simd_trace),
            bench_forest(&f3, simd_trace),
            bench_forest(&f4, simd_trace),
            bench_forest(&f5, simd_trace),
            bench_forest(&f6, simd_trace),
            bench_forest(&f7, simd_trace),
            bench_forest(&f8, simd_trace),
            bench_forest(&f9, simd_trace),
            bench_forest(&f10, simd_trace),
        ];

        for query_time in forest_results {
            write!(f_query, ",{}", query_time.as_secs_f64() / trace_len)?;
        }

        writeln!(f_query)?;
    }

    Ok(())
}

fn bench_forest<const T: usize>(
    forest: &PkdForest<3, T>,
    simd_trace: &[([f32x8; 3], f32x8)],
) -> Duration {
    stopwatch(|| {
        for (centers, radii) in simd_trace {
            black_box(forest.might_collide_simd(centers, radii * radii));
        }
    })
    .1
}
