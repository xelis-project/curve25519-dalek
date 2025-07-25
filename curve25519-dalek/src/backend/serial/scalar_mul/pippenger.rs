// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// Copyright (c) 2019 Oleg Andreev
// See LICENSE for licensing information.
//
// Authors:
// - Oleg Andreev <oleganza@gmail.com>

//! Implementation of a variant of Pippenger's algorithm.

#![allow(non_snake_case)]

use alloc::vec::Vec;

use core::borrow::Borrow;
use core::cmp::Ordering;

use crate::edwards::EdwardsPoint;
use crate::scalar::Scalar;
use crate::traits::VartimeMultiscalarMul;

/// Implements a version of Pippenger's algorithm.
///
/// The algorithm works as follows:
///
/// Let `n` be a number of point-scalar pairs.
/// Let `w` be a window of bits (6..8, chosen based on `n`, see cost factor).
///
/// 1. Prepare `2^(w-1) - 1` buckets with indices `[1..2^(w-1))` initialized with identity points.
///    Bucket 0 is not needed as it would contain points multiplied by 0.
/// 2. Convert scalars to a radix-`2^w` representation with signed digits in `[-2^w/2, 2^w/2]`.
///    Note: only the last digit may equal `2^w/2`.
/// 3. Starting with the last window, for each point `i=[0..n)` add it to a bucket indexed by
///    the point's scalar's value in the window.
/// 4. Once all points in a window are sorted into buckets, add buckets by multiplying each
///    by their index. Efficient way of doing it is to start with the last bucket and compute two sums:
///    intermediate sum from the last to the first, and the full sum made of all intermediate sums.
/// 5. Shift the resulting sum of buckets by `w` bits by using `w` doublings.
/// 6. Add to the return value.
/// 7. Repeat the loop.
///
/// Approximate cost w/o wNAF optimizations (A = addition, D = doubling):
///
/// ```ascii
/// cost = (n*A + 2*(2^w/2)*A + w*D + A)*256/w
///          |          |       |     |   |
///          |          |       |     |   looping over 256/w windows
///          |          |       |     adding to the result
///    sorting points   |       shifting the sum by w bits (to the next window, starting from last window)
///    one by one       |
///    into buckets     adding/subtracting all buckets
///                     multiplied by their indexes
///                     using a sum of intermediate sums
/// ```
///
/// For large `n`, dominant factor is (n*256/w) additions.
/// However, if `w` is too big and `n` is not too big, then `(2^w/2)*A` could dominate.
/// Therefore, the optimal choice of `w` grows slowly as `n` grows.
///
/// This algorithm is adapted from section 4 of <https://eprint.iacr.org/2012/549.pdf>.
pub struct Pippenger;

impl VartimeMultiscalarMul for Pippenger {
    type Point = EdwardsPoint;

    fn optional_multiscalar_mul<I, J>(scalars: I, points: J) -> Option<EdwardsPoint>
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator<Item = Option<EdwardsPoint>>,
    {
        use crate::traits::Identity;

        let mut scalars = scalars.into_iter();
        let size = scalars.by_ref().size_hint().0;

        // Digit width in bits. As digit width grows,
        // number of point additions goes down, but amount of
        // buckets and bucket additions grows exponentially.
        let w = if size < 500 {
            6
        } else if size < 800 {
            7
        } else {
            8
        };

        let max_digit: usize = 1 << w;
        let digits_count: usize = Scalar::to_radix_2w_size_hint(w);
        let buckets_count: usize = max_digit / 2; // digits are signed+centered hence 2^w/2, excluding 0-th bucket

        // Collect optimized scalars and points in buffers for repeated access
        // (scanning the whole set per digit position).
        let scalars = scalars.map(|s| s.borrow().as_radix_2w(w));

        let points = points
            .into_iter()
            .map(|p| p.map(|P| P.as_projective_niels()));

        let scalars_points = scalars
            .zip(points)
            .map(|(s, maybe_p)| maybe_p.map(|p| (s, p)))
            .collect::<Option<Vec<_>>>()?;

        // Prepare 2^w/2 buckets.
        // buckets[i] corresponds to a multiplication factor (i+1).
        let mut buckets: Vec<_> = (0..buckets_count)
            .map(|_| EdwardsPoint::identity())
            .collect();

        let mut columns = (0..digits_count).rev().map(|digit_index| {
            // Clear the buckets when processing another digit.
            for bucket in &mut buckets {
                *bucket = EdwardsPoint::identity();
            }

            // Iterate over pairs of (point, scalar)
            // and add/sub the point to the corresponding bucket.
            // Note: if we add support for precomputed lookup tables,
            // we'll be adding/subtracting point premultiplied by `digits[i]` to buckets[0].
            for (digits, pt) in scalars_points.iter() {
                // Widen digit so that we don't run into edge cases when w=8.
                let digit = digits[digit_index] as i16;
                match digit.cmp(&0) {
                    Ordering::Greater => {
                        let b = (digit - 1) as usize;
                        buckets[b] = (&buckets[b] + pt).as_extended();
                    }
                    Ordering::Less => {
                        let b = (-digit - 1) as usize;
                        buckets[b] = (&buckets[b] - pt).as_extended();
                    }
                    Ordering::Equal => {}
                }
            }

            // Add the buckets applying the multiplication factor to each bucket.
            // The most efficient way to do that is to have a single sum with two running sums:
            // an intermediate sum from last bucket to the first, and a sum of intermediate sums.
            //
            // For example, to add buckets 1*A, 2*B, 3*C we need to add these points:
            //   C
            //   C B
            //   C B A   Sum = C + (C+B) + (C+B+A)
            let mut buckets_intermediate_sum = buckets[buckets_count - 1];
            let mut buckets_sum = buckets[buckets_count - 1];
            for i in (0..(buckets_count - 1)).rev() {
                buckets_intermediate_sum += buckets[i];
                buckets_sum += buckets_intermediate_sum;
            }

            buckets_sum
        });

        // Take the high column as an initial value to avoid wasting time doubling the identity element in `fold()`.
        let hi_column = columns.next().expect("should have more than zero digits");

        Some(columns.fold(hi_column, |total, p| total.mul_by_pow_2(w as u32) + p))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::constants;

    #[test]
    fn test_vartime_pippenger() {
        // Reuse points across different tests
        let mut n = 512;
        let x = Scalar::from(2128506u64).invert();
        let y = Scalar::from(4443282u64).invert();
        let points: Vec<_> = (0..n)
            .map(|i| constants::ED25519_BASEPOINT_POINT * Scalar::from(1 + i as u64))
            .collect();
        let scalars: Vec<_> = (0..n)
            .map(|i| x + (Scalar::from(i as u64) * y)) // fast way to make ~random but deterministic scalars
            .collect();

        let premultiplied: Vec<EdwardsPoint> = scalars
            .iter()
            .zip(points.iter())
            .map(|(sc, pt)| sc * pt)
            .collect();

        while n > 0 {
            let scalars = &scalars[0..n].to_vec();
            let points = &points[0..n].to_vec();
            let control: EdwardsPoint = premultiplied[0..n].iter().sum();

            let subject = Pippenger::vartime_multiscalar_mul(scalars.clone(), points.clone());

            assert_eq!(subject.compress(), control.compress());

            n /= 2;
        }
    }
}
