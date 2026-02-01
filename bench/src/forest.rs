//! Power-of-two k-d forests.

use std::simd::{cmp::SimdPartialOrd, ptr::SimdConstPtr, Mask, Simd};

use crate::{distsq, median_partition};

#[derive(Clone, Debug)]
struct RandomizedTree<const K: usize> {
    tests: Box<[f32]>,
    seed: u32,
    points: Box<[[f32; K]]>,
}

#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct PkdForest<const K: usize, const T: usize> {
    test_seqs: [RandomizedTree<K>; T],
}

impl<const K: usize, const T: usize> PkdForest<K, T> {
    const T_NONE: Option<RandomizedTree<K>> = None;
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn new(points: &[[f32; K]]) -> Self {
        let mut trees = [Self::T_NONE; T];
        trees
            .iter_mut()
            .enumerate()
            .for_each(|(t, opt)| *opt = Some(RandomizedTree::new(points, t as u32)));
        Self {
            test_seqs: trees.map(Option::unwrap),
        }
    }

    #[must_use]
    /// # Panics
    ///
    /// This function will panic if `T` is 0.
    pub fn approx_nearest(&self, needle: [f32; K]) -> ([f32; K], f32) {
        self.test_seqs
            .iter()
            .map(|t| t.points[t.forward_pass(&needle)])
            .map(|point| (point, distsq(needle, point)))
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
            .unwrap()
    }

    #[must_use]
    pub fn might_collide(&self, needle: [f32; K], r_squared: f32) -> bool {
        self.test_seqs
            .iter()
            .any(|t| distsq(t.points[t.forward_pass(&needle)], needle) < r_squared)
    }

    #[must_use]
    pub fn might_collide_simd<const L: usize>(
        &self,
        needles: &[Simd<f32, L>; K],
        radii_squared: Simd<f32, L>,
    ) -> bool {
        let mut not_yet_collided = Mask::splat(true);

        for tree in &self.test_seqs {
            let indices = tree.mask_query(needles, not_yet_collided);
            let mut dists_sq = Simd::splat(0.0);
            let mut ptrs = Simd::splat(tree.points.as_ptr().cast()).wrapping_offset(indices);
            for needle_set in needles {
                let diffs =
                    unsafe { Simd::gather_select_ptr(ptrs, not_yet_collided, Simd::splat(0.0)) }
                        - needle_set;
                dists_sq += diffs * diffs;
                ptrs = ptrs.wrapping_add(Simd::splat(1));
            }

            not_yet_collided &= radii_squared.simd_lt(dists_sq).cast();

            if !not_yet_collided.all() {
                // at least one has collided - can return quickly
                return true;
            }
        }

        false
    }
}

impl<const K: usize> RandomizedTree<K> {
    pub fn new(points: &[[f32; K]], seed: u32) -> Self {
        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        fn recur_sort_points<const K: usize>(
            points: &mut [[f32; K]],
            tests: &mut [f32],
            test_dims: &mut [u8],
            i: usize,
            state: u32,
        ) {
            if points.len() > 1 {
                let d = state as usize % K;
                tests[i] = median_partition(points, d);
                test_dims[i] = u8::try_from(d).unwrap();
                let (lhs, rhs) = points.split_at_mut(points.len() / 2);
                recur_sort_points(lhs, tests, test_dims, 2 * i + 1, xorshift(state));
                recur_sort_points(rhs, tests, test_dims, 2 * i + 2, xorshift(state));
            }
        }

        assert!(K < u8::MAX as usize);

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut new_points = vec![[f32::INFINITY; K]; n2];
        new_points[..points.len()].copy_from_slice(points);
        let mut test_dims = vec![0; n2 - 1].into_boxed_slice();
        recur_sort_points(
            new_points.as_mut(),
            tests.as_mut(),
            test_dims.as_mut(),
            0,
            seed,
        );

        Self {
            tests,
            points: new_points.into_boxed_slice(),
            seed,
        }
    }

    fn forward_pass(&self, point: &[f32; K]) -> usize {
        let mut test_idx = 0;
        let mut k = 0;
        let mut state = self.seed;
        for _ in 0..self.tests.len().trailing_ones() {
            test_idx = 2 * test_idx
                + 1
                + usize::from(unsafe { *self.tests.get_unchecked(test_idx) } <= point[k]);
            state = xorshift(state);
            k = state as usize % K;
        }

        // retrieve affordance buffer location
        test_idx - self.tests.len()
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    /// Perform a masked SIMD query of this tree, only determining the location of the nearest
    /// neighbors for points in `mask`.
    fn mask_query<const L: usize>(
        &self,
        needles: &[Simd<f32, L>; K],
        mask: Mask<isize, L>,
    ) -> Simd<isize, L> {
        let mut test_idxs: Simd<isize, L> = Simd::splat(0);
        let mut state = self.seed;

        // Advance the tests forward
        for _ in 0..self.tests.len().trailing_ones() {
            let relevant_tests: Simd<f32, L> = unsafe {
                Simd::gather_select_ptr(
                    Simd::splat(self.tests.as_ptr().cast()).wrapping_offset(test_idxs),
                    mask,
                    Simd::splat(f32::NAN),
                )
            };
            let d = state as usize % K;
            let cmp_results: Mask<isize, L> = (needles[d].simd_ge(relevant_tests)).into();

            // TODO is there a faster way than using a conditional select?
            test_idxs <<= Simd::splat(1);
            test_idxs += Simd::splat(1);
            test_idxs += cmp_results.to_simd() & Simd::splat(1);
            state = xorshift(state);
        }

        test_idxs - Simd::splat(self.tests.len() as isize)
    }
}

#[inline]
/// Compute the next value in the xorshift sequence given the most recent value.
const fn xorshift(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_a_forest() {
        let points = [[0.0, 0.0], [0.2, 1.0], [-1.0, 0.4]];

        let forest = PkdForest::<2, 2>::new(&points);
        println!("{forest:#?}");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn find_the_closest() {
        let points = [[0.0, 0.0], [0.2, 1.0], [-1.0, 0.4]];

        let forest = PkdForest::<2, 2>::new(&points);
        // assert_eq!(forest.query1([0.01, 0.02]), ([]))
        let (nearest, ndsq) = forest.approx_nearest([0.01, 0.02]);
        assert_eq!(nearest, [0.0, 0.0]);
        assert!((ndsq - 0.0005) < 1e-6);
        println!("{:?}", forest.approx_nearest([0.01, 0.02]));
    }
}
