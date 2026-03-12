use bench::{forest::PkdForest, fuzz_pointcloud, get_points, make_needles};
use kiddo::SquaredEuclidean;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let mut starting_points = get_points(N);
    fuzz_pointcloud(&mut starting_points, 0.001, &mut rng);

    err_forest::<1>(&starting_points, &mut rng);
    err_forest::<2>(&starting_points, &mut rng);
    err_forest::<3>(&starting_points, &mut rng);
    err_forest::<4>(&starting_points, &mut rng);
    err_forest::<5>(&starting_points, &mut rng);
    err_forest::<6>(&starting_points, &mut rng);
    err_forest::<7>(&starting_points, &mut rng);
    err_forest::<8>(&starting_points, &mut rng);
    err_forest::<9>(&starting_points, &mut rng);
    err_forest::<10>(&starting_points, &mut rng);
}

fn err_forest<const T: usize>(points: &[[f32; 3]], rng: &mut impl Rng) {
    let forest = PkdForest::<3, T>::new(points);

    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in points {
        kiddo_kdt.add(pt, 0);
    }

    let (seq_needles, _) = make_needles::<3>(rng, 10_000);

    let mut total_err = 0.0;
    for &needle in &seq_needles {
        let (_, forest_distsq) = forest.approx_nearest(needle);
        let exact_distsq = kiddo_kdt.nearest_one::<SquaredEuclidean>(&needle).distance;

        let exact_dist = exact_distsq.sqrt();
        let err = forest_distsq.sqrt() - exact_dist;
        total_err += err;
        let rel_err = err / exact_distsq.sqrt();
        println!("{T}\t{err}\t{rel_err}\t{exact_dist}");
    }

    eprintln!("T={T}: mean error {}", total_err / seq_needles.len() as f32);
}
