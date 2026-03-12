use std::mem::size_of;

use crate::{distsq, forward_pass, median_partition};
use capt::Aabb;
use wide::{f32x8, i32x8, CmpGe, CmpLt};

#[derive(Clone, Debug, PartialEq)]
/// A power-of-two KD-tree.
///
/// # Generic parameters
///
/// - `D`: The dimension of the space.
pub struct PkdTree<const K: usize> {
    /// The test values for determining which part of the tree to enter.
    ///
    /// The first element of `tests` should be the first value to test against.
    /// If we are less than `tests[0]`, we move on to `tests[1]`; if not, we move on to `tests[2]`.
    /// At the `i`-th test performed in sequence of the traversal, if we are less than
    /// `tests[idx]`, we advance to `2 * idx + 1`; otherwise, we go to `2 * idx + 2`.
    ///
    /// The length of `tests` must be `N`, rounded up to the next power of 2, minus one.
    tests: Box<[f32]>,
    /// The relevant points at the center of each volume divided by `tests`.
    points: Box<[[f32; K]]>,
}

impl<const K: usize> PkdTree<K> {
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    /// Construct a new `PkdTree` containing all the points in `points`.
    /// For performance, this function changes the ordering of `points`, but does not affect the
    /// set of points inside it.
    ///
    /// # Panics
    ///
    /// This function will panic if `D` is greater than or equal to 255.
    ///
    /// TODO: do all our sorting on the allocation that we return?
    pub fn new(points: &[[f32; K]]) -> Self {
        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        /// Runs in O(n log n)
        fn build_tree<const K: usize>(points: &mut [[f32; K]], tests: &mut [f32], k: u8, i: usize) {
            if points.len() > 1 {
                tests[i] = median_partition(points, k as usize);
                let next_k = (k + 1) % K as u8;
                let (lhs, rhs) = points.split_at_mut(points.len() / 2);
                build_tree(lhs, tests, next_k, 2 * i + 1);
                build_tree(rhs, tests, next_k, 2 * i + 2);
            }
        }

        assert!(K < u8::MAX as usize);

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut new_points = vec![[f32::INFINITY; K]; n2].into_boxed_slice();
        new_points[..points.len()].copy_from_slice(points);
        build_tree(new_points.as_mut(), tests.as_mut(), 0, 0);

        Self {
            tests,
            points: new_points,
        }
    }

    #[must_use]
    pub fn approx_nearest(&self, needle: [f32; K]) -> [f32; K] {
        self.get_point(forward_pass(&self.tests, &needle))
    }

    #[must_use]
    /// Determine whether a ball centered at `needle` with radius `r_squared` could collide with a
    /// point in this tree.
    pub fn might_collide(&self, needle: [f32; K], r_squared: f32) -> bool {
        distsq(self.approx_nearest(needle), needle) <= r_squared
    }

    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn might_collide_simd(&self, needles: &[f32x8; K], radii_squared: f32x8) -> bool {
        let indices = forward_pass_wide(&self.tests, needles);
        let idx_arr = indices.to_array();
        let mut dists_squared = f32x8::ZERO;
        for (k, needle_values) in needles.iter().enumerate() {
            let vals = f32x8::new(idx_arr.map(|i| self.points[i as usize][k]));
            let deltas = vals - needle_values;
            dists_squared = dists_squared + deltas * deltas;
        }
        dists_squared.simd_lt(radii_squared).any()
    }

    #[must_use]
    #[allow(clippy::cast_possible_wrap, clippy::missing_panics_doc)]
    pub fn query1_exact(&self, needle: [f32; K]) -> usize {
        let mut id = usize::MAX;
        let mut best_distsq = f32::INFINITY;
        self.exact_help(
            0,
            0,
            &Aabb {
                lo: [-f32::INFINITY; K],
                hi: [f32::INFINITY; K],
            },
            needle,
            &mut id,
            &mut best_distsq,
        );
        id
    }

    #[allow(clippy::cast_possible_truncation)]
    fn exact_help(
        &self,
        test_idx: usize,
        k: u8,
        bounding_box: &Aabb<f32, K>,
        point: [f32; K],
        best_id: &mut usize,
        best_distsq: &mut f32,
    ) {
        if bounding_box.closest_distsq_to(&point) > *best_distsq {
            return;
        }

        if self.tests.len() <= test_idx {
            let id = test_idx - self.tests.len();
            let new_distsq = distsq(point, self.get_point(id));
            if new_distsq < *best_distsq {
                *best_id = id;
                *best_distsq = new_distsq;
            }

            return;
        }

        let test = self.tests[test_idx];

        let mut bb_below = *bounding_box;
        bb_below.hi[k as usize] = test;
        let mut bb_above = *bounding_box;
        bb_above.lo[k as usize] = test;

        let next_k = (k + 1) % K as u8;
        if point[k as usize] < test {
            self.exact_help(
                2 * test_idx + 1,
                next_k,
                &bb_below,
                point,
                best_id,
                best_distsq,
            );
            self.exact_help(
                2 * test_idx + 2,
                next_k,
                &bb_above,
                point,
                best_id,
                best_distsq,
            );
        } else {
            self.exact_help(
                2 * test_idx + 2,
                next_k,
                &bb_above,
                point,
                best_id,
                best_distsq,
            );
            self.exact_help(
                2 * test_idx + 1,
                next_k,
                &bb_below,
                point,
                best_id,
                best_distsq,
            );
        }
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn get_point(&self, id: usize) -> [f32; K] {
        self.points[id]
    }

    #[must_use]
    /// Return the total memory used (stack + heap) by this structure.
    pub fn memory_used(&self) -> usize {
        size_of::<Self>() + (self.points.len() * K + self.tests.len()) * size_of::<f32>()
    }
}

#[inline]
#[allow(clippy::cast_sign_loss)]
fn forward_pass_wide<const K: usize>(tests: &[f32], centers: &[f32x8; K]) -> i32x8 {
    let mut test_idxs = i32x8::splat(0_i32);
    let mut k = 0;
    for _ in 0..tests.len().trailing_ones() {
        let idx_arr = test_idxs.to_array();
        let relevant_tests =
            f32x8::new(idx_arr.map(|i| unsafe { *tests.get_unchecked(i as usize) }));
        let cmp_f = centers[k % K].simd_ge(relevant_tests);
        let cmp_bit: i32x8 =
            unsafe { core::mem::transmute::<f32x8, i32x8>(cmp_f) } & i32x8::splat(1);
        test_idxs = (test_idxs << 1_i32) + 1_i32 + cmp_bit;
        k = (k + 1) % K;
    }
    test_idxs - i32x8::splat(tests.len() as i32)
}

#[cfg(test)]
mod tests {

    use crate::forward_pass;

    use super::*;

    #[test]
    fn single_query() {
        let points = vec![
            [0.1, 0.1],
            [0.1, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.2],
            [0.7, 0.8],
        ];
        let kdt = PkdTree::new(&points);

        println!("testing for correctness...");

        let neg1 = [-1.0, -1.0];
        let neg1_idx = forward_pass(&kdt.tests, &neg1);
        assert_eq!(neg1_idx, 0);

        let pos1 = [1.0, 1.0];
        let pos1_idx = forward_pass(&kdt.tests, &pos1);
        assert_eq!(pos1_idx, points.len() - 1);
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn multi_query() {
        let points = vec![
            [0.1, 0.1],
            [0.1, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.2],
            [0.7, 0.8],
        ];
        let kdt = PkdTree::new(&points);

        let needles = [
            f32x8::new([-1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            f32x8::new([-1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let result = forward_pass_wide(&kdt.tests, &needles).to_array();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], (points.len() - 1) as i32);
    }

    #[test]
    fn not_a_power_of_two() {
        let points = vec![[0.0], [2.0], [4.0]];
        let kdt = PkdTree::new(&points);

        println!("{kdt:?}");

        assert_eq!(forward_pass(&kdt.tests, &[-0.1]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[0.5]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[1.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[2.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[3.5]), 2);
        assert_eq!(forward_pass(&kdt.tests, &[4.5]), 2);
    }

    #[test]
    fn a_power_of_two() {
        let points = vec![[0.0], [2.0], [4.0], [6.0]];
        let kdt = PkdTree::new(&points);

        println!("{kdt:?}");

        assert_eq!(forward_pass(&kdt.tests, &[-0.1]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[0.5]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[1.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[2.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[3.5]), 2);
        assert_eq!(forward_pass(&kdt.tests, &[4.5]), 2);
    }
}
