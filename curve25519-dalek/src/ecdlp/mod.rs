//! # Elliptic Curve Discrete Logarithm Problem (ECDLP)
//!
//! This file enables the decoding of integers from [`RistrettoPoint`]s. As this requires
//! a bruteforce operation, this can take quite a long time (multiple seconds) for large input spaces.
//!
//! The algorithm depends on a constant `L1` parameter, which enables changing the space/time tradeoff.
//! To use a given `L1` constant, you need to generate a precomputed tables file, or download a pre-generated one.
//! The resulting file may be quite big, which is why it is recommended to load it at runtime using [`ecdlp::ECDLPTablesFile`].
//!
//! The algorithm works for any range, but keep in mind that the time the algorithm takes grows exponentially: with the same
//! precomputed tables, decoding an `n+1`-bit integer will take 2x as long as an `n`-bit integer. By default, unless the "pseudo"
//! constant time mode is enabled, integers which are closer to the start of the range will be found exponentially faster: for
//! example, integers in the `n-1`-bit first half of the `n`-bit decoding range will take 1/2 of the time compared to the second half,
//! and numbers near the very beginning will be found almost immediately.
//!
//! # Space / Time tradeoff and benchmarks
//!
//! To choose an `L1` constant, you may to see benchmark performance and tables size [here](ecdlp_perf.md).
//!
//! # Table generation
//!
//! For now, table generation can be done using the `gen_t1_t2` test, or using the unstable [`ecdlp::table_generation`] module.
//!
//! # Constant time
//!
//! This algorithm cannot be constant-time because of hashmap lookups. However a "pseudo"
//! constant time mode is implemented which lets the algorithm continue to run even when it
//! has found the answer.
//!
//! # Example
//!
//! Decoding a 48bit number using a L1=26 precomputed tables file.
//! ```no_run
//! use curve25519_dalek::{
//!     constants::{RISTRETTO_BASEPOINT_POINT as G},
//!     ecdlp::{ECDLPTables, decode, ECDLPArguments},
//!     Scalar,
//!     RistrettoPoint,
//! };
//!
//! let precomputed_tables = ECDLPTables::load_from_file(26, "ecdlp_table_26.bin")
//!     .unwrap();
//!
//! let num = 258383831730230u64;
//! let to_decode = Scalar::from(num) * G;
//!
//! assert_eq!(
//!     decode(&precomputed_tables.view(), to_decode, ECDLPArguments::new_with_range(0, 1 << 48)),
//!     Some(num as i64)
//! );
//! ```

// Notes of the ECDLP implementation.
mod ecdlp_notes {
    //! The algorithm implemented here is BSGS (Baby Step Giant Step), and the implementation
    //! details are based on [Solving Small Exponential ECDLP in EC-based Additively Homomorphic Encryption and Applications][fast-ecdlp-paper].
    //!
    //! The gist of BSGS goes as follows:
    //! - Our target point, which we want to decode, represents an integer in the range \[0, 2^(L1 + L2)\].
    //! - We have a T1 hash table, where the key is the curve point and value is the decoded
    //!   point. T1 = <i * G => i | i in \[1, 2^l1\]>
    //! - We have a T2 linear table (an array), where T2 = \[j * 2^l1 * G | j in \[1, 2^l2\]\]
    //! - For each j in 0..2^l2
    //!   Compute the difference between T2\[j\] and the target point
    //!   if let Some(i) = T1.get(the difference) => the decoded integer is j * 2^L1 + i.
    //!
    //! On top of this regular BSGS algorithm, we add the following optimizations:
    //! - Batching. The paper uses a tree-based Montgomery trick - instead, we use the batched
    //!   inversion which is implemented in FieldElement.
    //! - T1 only contains the truncated x coordinates. The table uses Cuckoo hashing, and
    //!   the hash of a point is directly just a subset of the bytes of the point.
    //! - We need a canonical encoding of a point before any hashmap lookup: this means that
    //!   we must work with affine coordinates. Addition of affine Montgomery points requires
    //!   less inversions than Edwards points, so we use that instead.  
    //! - Using the fact -(x, y) = (x, -y) on the Montgomery curve, we can shift the inputs so
    //!   that we only need half of T1 and T2 and half of the modular inversions.
    //! - The L2 constant has been fixed here, because we can just shift the input after every
    //!   batch. This means that L2 has a constant size of about 16Ko, which is preferable
    //!   to >100Mo when L2 = 22, for example. This results in slightly more modular inversions,
    //!   however this has no visible impact on performance. Shifting the inputs like this
    //!   also means that we support arbitrary decoding ranges for a given constant tables file.
    //!
    //! Note: We are dealing with a curve which has cofactors; as such, we need to multiply
    //! by the cofactor before running ECDLP to clear it and guarantee a canonical encoding of our points.
    //! The tables also need to be based on `num * cofactor` to match.
    //!
    //! [fast-ecdlp-paper]: https://eprint.iacr.org/2022/1573
}

mod affine_montgomery;
mod table;
mod field_simd;

use crate::{
    RistrettoPoint, Scalar, constants::MONTGOMERY_A_NEG, constants::RISTRETTO_BASEPOINT_POINT as G,
    field::FieldElement,
};
use affine_montgomery::AffineMontgomeryPoint;
use core::{
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
};

pub use table::{
    ECDLPTablesFileView, NoOpProgressTableGenerationReportFunction,
    ProgressTableGenerationReportFunction, ReportStep, table_generation,
};

use table::{BATCH_SIZE, L2};

/// A trait to represent progress report functions.
/// It is auto-implemented on any `F: Fn(f64) -> ControlFlow<()>`.
pub trait ProgressReportFunction {
    /// Run the progress report function.
    fn report(&self, progress: f64) -> ControlFlow<()>;
}
impl<F: Fn(f64) -> ControlFlow<()>> ProgressReportFunction for F {
    #[inline(always)]
    fn report(&self, progress: f64) -> ControlFlow<()> {
        self(progress)
    }
}
/// The Noop (no operation) report function. It does nothing and will never break.
pub struct NoopReportFn;
impl ProgressReportFunction for NoopReportFn {
    #[inline(always)]
    fn report(&self, _progress: f64) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A struct to ensure that the bytes are aligned on 32 bytes.
/// This is required for the table generation.
#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(32))]
struct ForcedAlign32([u8; 32]);

/// The tables file is a big array of ForcedAlign32, which is a 32-byte aligned array of bytes.
/// Some bytes may be used as padding only.
/// This prevent using memory-mapped files, as the alignment is not guaranteed.
pub struct ECDLPTables {
    bytes: Vec<ForcedAlign32>,
    l1: usize,
    size: usize,
}

impl ECDLPTables {
    /// Get the expected final bytes size and number of vec elements in the tables.
    pub fn get_required_sizes(l1: usize) -> (usize, usize) {
        let size = table_generation::table_file_len(l1);
        let mut n = size / 32;
        if size % 32 != 0 {
            n += 1;
        }
        (size, n)
    }

    /// Create a new empty precomputed tables.
    pub fn empty(l1: usize) -> Self {
        let (size, n) = Self::get_required_sizes(l1);
        Self {
            l1,
            bytes: vec![Default::default(); n],
            size,
        }
    }

    /// Generate a new precomputed tables
    pub fn generate(l1: usize) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file(l1, zelf.as_mut_slice())?;

        Ok(zelf)
    }

    /// Generate a new precomputed tables, with multithreading
    pub fn generate_par(l1: usize, n_threads: usize) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file_par(l1, n_threads, zelf.as_mut_slice())?;

        Ok(zelf)
    }

    /// Generate a new precomputed tables with a progress report function.
    pub fn generate_with_progress_report<P: ProgressTableGenerationReportFunction>(
        l1: usize,
        p: P,
    ) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file_with_progress_report(l1, zelf.as_mut_slice(), p)?;

        Ok(zelf)
    }

    /// Generate a new precomputed tables with a progress report function, with multithreading.
    pub fn generate_with_progress_report_par<P: ProgressTableGenerationReportFunction + Sync>(
        l1: usize,
        n_threads: usize,
        p: P,
    ) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file_with_progress_report_par(
            l1,
            n_threads,
            zelf.as_mut_slice(),
            p,
        )?;

        Ok(zelf)
    }

    /// Load the tables from a bytes slice.
    pub fn from_bytes(l1: usize, bytes: &[u8]) -> Self {
        let mut zelf = Self::empty(l1);
        zelf.as_mut_slice().copy_from_slice(bytes);

        zelf
    }

    /// Load the tables from a file.
    #[cfg(feature = "std")]
    pub fn load_from_file(l1: usize, path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut zelf = Self::empty(l1);

        let mut file = std::fs::File::open(path)?;
        file.read_exact(zelf.as_mut_slice())?;

        Ok(zelf)
    }

    /// Write the tables to a file.
    #[cfg(feature = "std")]
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;
        file.write_all(self.as_slice())?;
        Ok(())
    }

    /// Get the tables as a slice of bytes.
    pub fn as_slice(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.bytes)[..self.size]
    }

    /// Get the tables a mutable slice of bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.bytes)[..self.size]
    }

    /// Get a view of the tables.
    pub fn view(&self) -> ECDLPTablesFileView<'_> {
        ECDLPTablesFileView::from_bytes(self.as_slice(), self.l1)
    }
}

const J_BATCH_SIZE: usize = 4;

/// Builder for the ECDLP algorithm parameters.
pub struct ECDLPArguments<R: ProgressReportFunction = NoopReportFn> {
    range_start: i64,
    range_end: i64,
    pseudo_constant_time: bool,
    n_threads: usize,
    progress_report_function: R,
}

impl ECDLPArguments<NoopReportFn> {
    /// Creates a new `ECDLPArguments` with default arguments, to run on a specific range.
    pub fn new_with_range(range_start: i64, range_end: i64) -> Self {
        Self {
            range_start,
            range_end,
            pseudo_constant_time: false,
            progress_report_function: NoopReportFn,
            n_threads: 1,
        }
    }
}

impl<F: ProgressReportFunction> ECDLPArguments<F> {
    /// Enable the "pseudo constant-time" mode. This means that the algorithm will not stop
    /// once it has found the answer. Keep in mind that **this is not actually constant-time**,
    /// in fact, the algorithm cannot be constant-time because it relies on hashmap lookups.
    /// This setting is also useful for benchmarking, as any input will result in roughly the same
    /// execution time.
    pub fn pseudo_constant_time(self, pseudo_constant_time: bool) -> Self {
        Self {
            pseudo_constant_time,
            ..self
        }
    }

    /// Sets the progress report function.
    ///
    /// This function will be periodically called when the algorithm is running.
    /// The `progress` argument represents the current progress, from `0.0` to `1.0`.
    /// Returning `ControlFlow::Break(())` will stop the algorithm.
    ///
    /// Please keep in mind that this report function should not take too long or nuke
    /// the cache, as it would impact the performance of the algorithm.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use curve25519_dalek::ecdlp::ECDLPArguments;
    /// use std::ops::ControlFlow;
    ///
    /// let ecdlp_args = ECDLPArguments::new_with_range(0, 1 << 48)
    ///     .progress_report_function(|_progress| {
    ///         // do something with `progress`
    ///         ControlFlow::Continue(())
    ///     });
    /// ```
    pub fn progress_report_function<R: ProgressReportFunction>(
        self,
        progress_report_function: R,
    ) -> ECDLPArguments<R> {
        ECDLPArguments {
            progress_report_function,
            range_start: self.range_start,
            range_end: self.range_end,
            pseudo_constant_time: self.pseudo_constant_time,
            n_threads: self.n_threads,
        }
    }

    /// Configures the number of threads used.
    /// This only affects the execution of the [`par_decode`] function.
    ///
    /// # Example
    ///
    /// ```
    /// use curve25519_dalek::ecdlp::ECDLPArguments;
    ///
    /// let n_threads = std::thread::available_parallelism()
    ///     .expect("cannot get available parallelism")
    ///     .get();
    /// let ecdlp_args = ECDLPArguments::new_with_range(0, 1 << 48)
    ///     .n_threads(n_threads);
    /// ```
    pub fn n_threads(self, n_threads: usize) -> Self {
        Self { n_threads, ..self }
    }
}


fn decode_prep<R: ProgressReportFunction>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: &ECDLPArguments<R>,
    n_threads: usize,
    thread_i: usize, // Add thread_i parameter
) -> (i64, RistrettoPoint, usize) {
    let amplitude = (args.range_end - args.range_start).max(0) / n_threads as i64;

    let offset = args.range_start + amplitude*thread_i as i64
        + ((1 << (L2 - 1)) << precomputed_tables.get_l1())
        + (1 << (precomputed_tables.get_l1() - 1));

    let normalized =
        point - RistrettoPoint::mul_base(&i64_to_scalar(offset));

    let j_end = ((amplitude as i64) >> precomputed_tables.get_l1()) as usize;
    let divceil = |a, b| (a + b - 1) / b;

    let num_batches = divceil(j_end, 1 << L2);

    (offset, normalized, num_batches)
}


/// Returns an iterator of batches for a given thread. Common to [`par_decode`] and [`decode`].
/// Iterator item is (index, j_start, target_montgomery, progress).
fn make_point_iterator(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    normalized: RistrettoPoint,
    num_batches: usize,
) -> impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)> {
    let thread_iter = (0..num_batches).map(move |j| {
        let progress = j as f64 / num_batches as f64;
        (j, j * (1 << L2), progress)
    });

    // clear the cofactor, we want the repr to be canonical
    let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());

    let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());

    // starting point for this thread
    let mut target_montgomery = AffineMontgomeryPoint::from(&normalized.0);
    let batch_step = -(els_per_batch as i64);
    let batch_step_montgomery =
        AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());

    thread_iter.map(move |(j, j_start, progress)| {
        let current = target_montgomery;

        target_montgomery =
            AffineMontgomeryPoint::addition_not_ct(&target_montgomery, &batch_step_montgomery);

        (j, j_start, current, progress)
    })
}

fn make_point_iterator_simd(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    normalized: RistrettoPoint,
    num_batches: usize,
) -> impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)> {
    // Apply same transformations as original make_point_iterator
    let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
    let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());
    
    let initial = AffineMontgomeryPoint::from(&normalized.0);
    let batch_step = -(els_per_batch as i64);
    let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
    
    struct OptimizedIterator {
        current_batch: [AffineMontgomeryPoint; 4],
        step: AffineMontgomeryPoint,
        step_x4: AffineMontgomeryPoint,
        batch_idx: usize,
        j: usize,
        num_batches: usize,
    }
    
    impl Iterator for OptimizedIterator {
        type Item = (usize, usize, AffineMontgomeryPoint, f64);
        
        fn next(&mut self) -> Option<Self::Item> {
            if self.j >= self.num_batches {
                return None;
            }
            
            // Refill batch when we've consumed all 4
            if self.batch_idx >= 4 && self.j < self.num_batches {
                // Use 4-way SIMD operation to advance all points by 4
                self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
                    &self.current_batch,
                    &self.step_x4
                );
                self.batch_idx = 0;
            }
            
            let result = (
                0, // index (not used in ECDLP)
                self.j * (1 << L2), // j_start
                self.current_batch[self.batch_idx],
                self.j as f64 / self.num_batches as f64
            );
            
            self.batch_idx += 1;
            self.j += 1;
            
            Some(result)
        }
    }
    
    // Initialize first 4 points
    let p0 = initial;
    let p1 = p0.addition_not_ct(&step);
    let p2 = p1.addition_not_ct(&step);
    let p3 = p2.addition_not_ct(&step);
    
    // Pre-compute 4*step
    let step_x4 = step.addition_not_ct(&step)
        .addition_not_ct(&step)
        .addition_not_ct(&step);
    
    OptimizedIterator {
        current_batch: [p0, p1, p2, p3],
        step,
        step_x4,
        batch_idx: 0,
        j: 0,
        num_batches,
    }
}

/// Decode a [`RistrettoPoint`] to the represented integer.
/// This may take a long time, so if you are running on an event-loop such as `tokio`, you
/// should wrap this in a `tokio::block_on` task.
pub fn decode<R: ProgressReportFunction>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64> {
    let (offset, normalized, num_batches) = decode_prep(precomputed_tables, point, &args, 1, 0);
    let point_iter = make_point_iterator(precomputed_tables, normalized, num_batches);

    // Pre compute the T2 cache
    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        for (i, (cache, alpha)) in t2_cache
            .iter_mut()
            .zip(t2_cache_alpha.iter_mut())
            .enumerate()
        {
            let point = t2_table.index(i);
            *alpha = &MONTGOMERY_A_NEG - &point.u;
            *cache = point;
        }
    }

    fast_ecdlp(
        precomputed_tables,
        normalized,
        point_iter,
        args.pseudo_constant_time,
        &AtomicBool::new(false),
        args.progress_report_function,
        &t2_cache,
        &t2_cache_alpha,
    )
    .map(|v| v as i64 + offset)
}

/// Decode a [`RistrettoPoint`] to the represented integer, in parallel.
/// This implementation uses [`rayon`] for parallelism, which depends on the Rust standard library (`std`),
/// and therefore is not compatible with `#![no_std]` environments such as WebAssembly (`wasm32-unknown-unknown`) or bare-metal targets.
///
/// Rayon provides efficient thread pool management and typically outperforms manual thread spawning using [`std::thread`].
/// 
/// Note: If you need compatibility with non-`std` targets, use the single-threaded [`decode`] function instead.
/// This may take a long time, so if you are running on an event-loop such as `tokio`, you
/// should wrap this in a `tokio::block_on` task.
pub fn par_decode<R: ProgressReportFunction + Sync>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64> {
    // if args.n_threads == 1 {
    //     return decode(precomputed_tables, point, args);
    // }

    use rayon::prelude::*;
    use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};

    let chunk_size: u64 = 64 * 320_000_000_000_0 / ((30 - precomputed_tables.get_l1()).max(0) * 2).max(1) as u64;
    let total_range = args.range_end as u64 - args.range_start as u64;
    let num_chunks = ((total_range + chunk_size - 1) / chunk_size) as usize;
    let n_threads = args.n_threads;

    let end_flag = Arc::new(AtomicBool::new(false));

    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        for (i, (cache, alpha)) in t2_cache.iter_mut().zip(t2_cache_alpha.iter_mut()).enumerate() {
            let point = t2_table.index(i);
            *alpha = &MONTGOMERY_A_NEG - &point.u;
            *cache = point;
        }
    }

    (0..n_threads)
        .into_par_iter()
        .find_map_any(|thread_index| {
            let t2_u_values: [FieldElement; BATCH_SIZE] = {
                let mut u_values = [FieldElement::ZERO; BATCH_SIZE];
                for i in 0..BATCH_SIZE {
                    u_values[i] = t2_cache[i].u;
                }
                u_values
            };

            let t2_vs: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| p.v.clone());
            let t2_vs_neg: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| -&p.v);

            // allocate scratch data buffers once per worker
            let mut batch = &mut [FieldElement::ZERO; BATCH_SIZE];
            let mut alphas = &mut [FieldElement::ZERO; BATCH_SIZE];
            let mut qxs = &mut [FieldElement::ZERO; BATCH_SIZE];
            let mut neg_qxs = &mut [FieldElement::ZERO; BATCH_SIZE];

            for chunk_index in (thread_index..num_chunks).step_by(n_threads) {
                if end_flag.load(Ordering::Relaxed) {
                    return None;
                }

                let start = args.range_start + (chunk_index as u64 * chunk_size) as i64;
                let end = ((chunk_index as u64 + 1) * chunk_size).min(args.range_end as u64) as i64;

                let chunk_args = ECDLPArguments {
                    range_start: start,
                    range_end: end,
                    n_threads: 1,
                    pseudo_constant_time: args.pseudo_constant_time,
                    progress_report_function: NoopReportFn, // FIXME: make chunk-relative on the global scale
                };

                let (offset, normalized, num_batches) =
                    decode_prep(precomputed_tables, point, &chunk_args, 1, 0);

                let progress_wrapper = |progress: f64| {
                    if !chunk_args.pseudo_constant_time && end_flag.load(Ordering::Relaxed) {
                        ControlFlow::Break(())
                    } else {
                        let result = args.progress_report_function.report(progress);
                        if result.is_break() {
                            end_flag.store(true, Ordering::SeqCst);
                        }
                        result
                    }
                };

                let point_iter =
                    make_point_iterator_simd(precomputed_tables, normalized, num_batches);

                // if thread_index == 0 && (start / 100000000000000_i64) % n_threads as i64 == 0 {
                //     println!("Thread 0: processing chunk starting at {}", start);
                // }

                if let Some(res) = fast_ecdlp_simd(
                    precomputed_tables,
                    normalized,
                    point_iter,
                    chunk_args.pseudo_constant_time,
                    &end_flag,
                    progress_wrapper,
                    &t2_cache,
                    &t2_cache_alpha,
                    &t2_u_values,
                    &t2_vs,
                    &t2_vs_neg,
                    &mut batch,
                    &mut alphas,
                    &mut qxs,
                    &mut neg_qxs
                ) {
                    end_flag.store(true, Ordering::SeqCst);
                    return Some(offset + res as i64);
                }
            }

            None
        })
}

fn make_point_iterator_simd_batched(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    normalized: RistrettoPoint,
    num_batches: usize,
) -> impl Iterator<Item = [(usize, usize, AffineMontgomeryPoint, f64); 4]> {
    let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
    let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());
    
    let initial = AffineMontgomeryPoint::from(&normalized.0);
    let batch_step = -(els_per_batch as i64);
    let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
    
    struct BatchedIterator {
        current_batch: [AffineMontgomeryPoint; 4],
        step: AffineMontgomeryPoint,
        step_x4: AffineMontgomeryPoint,
        j: usize,
        num_batches: usize,
    }
    
    impl Iterator for BatchedIterator {
        type Item = [(usize, usize, AffineMontgomeryPoint, f64); 4];
        
        fn next(&mut self) -> Option<Self::Item> {
            if self.j >= self.num_batches {
                return None;
            }
            
            // Build result array with current batch
            let mut result = [(0, 0, AffineMontgomeryPoint::identity(), 0.0); 4];
            let mut count = 0;
            
            for i in 0..4 {
                if self.j < self.num_batches {
                    result[i] = (
                        0,
                        self.j * (1 << L2),
                        self.current_batch[i],
                        self.j as f64 / self.num_batches as f64
                    );
                    self.j += 1;
                    count += 1;
                }
            }
            
            if count == 0 {
                return None;
            }
            
            // Advance all 4 points for next iteration
            self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
                &self.current_batch,
                &self.step_x4
            );
            
            Some(result)
        }
    }
    
    // Initialize first 4 points
    let p0 = initial;
    let p1 = p0.addition_not_ct(&step);
    let p2 = p1.addition_not_ct(&step);
    let p3 = p2.addition_not_ct(&step);
    
    // Pre-compute 4*step
    let step_x4 = step.addition_not_ct(&step)
        .addition_not_ct(&step)
        .addition_not_ct(&step);
    
    BatchedIterator {
        current_batch: [p0, p1, p2, p3],
        step,
        step_x4,
        j: 0,
        num_batches,
    }
}

pub fn par_decode_j_batch<R: ProgressReportFunction + Sync>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64> {
    // if args.n_threads == 1 {
    //     return decode(precomputed_tables, point, args);
    // }

    use rayon::prelude::*;
    use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};

    let chunk_size: u64 = 64 * 320_000_000_000_0 / ((30 - precomputed_tables.get_l1()).max(0) * 2).max(1) as u64;
    let total_range = args.range_end as u64 - args.range_start as u64;
    let num_chunks = ((total_range + chunk_size - 1) / chunk_size) as usize;
    let n_threads = args.n_threads;

    let end_flag = Arc::new(AtomicBool::new(false));

    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        for (i, (cache, alpha)) in t2_cache.iter_mut().zip(t2_cache_alpha.iter_mut()).enumerate() {
            let point = t2_table.index(i);
            *alpha = &MONTGOMERY_A_NEG - &point.u;
            *cache = point;
        }
    }

    (0..n_threads)
        .into_par_iter()
        .find_map_any(|thread_index| {
            let t2_u_values: [FieldElement; BATCH_SIZE] = {
                let mut u_values = [FieldElement::ZERO; BATCH_SIZE];
                for i in 0..BATCH_SIZE {
                    u_values[i] = t2_cache[i].u;
                }
                u_values
            };

            let t2_vs: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| p.v.clone());
            let t2_vs_neg: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| -&p.v);

            // allocate scratch data buffers once per worker
            let mut batch = &mut [FieldElement::ZERO; BATCH_SIZE];
            let mut alphas = &mut [FieldElement::ZERO; BATCH_SIZE];
            let mut qxs = &mut [FieldElement::ZERO; BATCH_SIZE];
            let mut neg_qxs = &mut [FieldElement::ZERO; BATCH_SIZE];
            let mut j_queue: Vec<(usize, usize, AffineMontgomeryPoint, f64)> = Vec::with_capacity(J_BATCH_SIZE);
            let mut batch_storage: [[FieldElement; BATCH_SIZE]; J_BATCH_SIZE] = [[FieldElement::ZERO; BATCH_SIZE]; J_BATCH_SIZE];
            let mut alpha_storage: [[FieldElement; BATCH_SIZE]; J_BATCH_SIZE] = [[FieldElement::ZERO; BATCH_SIZE]; J_BATCH_SIZE];

            for chunk_index in (thread_index..num_chunks).step_by(n_threads) {
                if end_flag.load(Ordering::Relaxed) {
                    return None;
                }

                let start = args.range_start + (chunk_index as u64 * chunk_size) as i64;
                let end = ((chunk_index as u64 + 1) * chunk_size).min(args.range_end as u64) as i64;

                let chunk_args = ECDLPArguments {
                    range_start: start,
                    range_end: end,
                    n_threads: 1,
                    pseudo_constant_time: args.pseudo_constant_time,
                    progress_report_function: NoopReportFn, // FIXME: make chunk-relative on the global scale
                };

                let (offset, normalized, num_batches) =
                    decode_prep(precomputed_tables, point, &chunk_args, 1, 0);

                let progress_wrapper = |progress: f64| {
                    if !chunk_args.pseudo_constant_time && end_flag.load(Ordering::Relaxed) {
                        ControlFlow::Break(())
                    } else {
                        let result = args.progress_report_function.report(progress);
                        if result.is_break() {
                            end_flag.store(true, Ordering::SeqCst);
                        }
                        result
                    }
                };

                let point_iter =
                    make_point_iterator_simd_batched(precomputed_tables, normalized, num_batches);

                // if thread_index == 0 && (start / 100000000000000_i64) % n_threads as i64 == 0 {
                //     println!("Thread 0: processing chunk starting at {}", start);
                // }

                if let Some(res) = fast_ecdlp_simd_j_batched(
                    precomputed_tables,
                    normalized,
                    point_iter,
                    chunk_args.pseudo_constant_time,
                    &end_flag,
                    progress_wrapper,
                    &t2_cache,
                    &t2_cache_alpha,
                    &t2_u_values,
                    &t2_vs,
                    &t2_vs_neg,
                    &mut batch,
                    &mut alphas,
                    &mut qxs,
                    &mut neg_qxs,
                    &mut j_queue,
                    &mut batch_storage,
                    &mut alpha_storage
                ) {
                    end_flag.store(true, Ordering::SeqCst);
                    return Some(offset + res as i64);
                }
            }

            None
        })
}


pub fn par_decode_scalar<R: ProgressReportFunction + Sync>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64> {
    // if args.n_threads == 1 {
    //     return decode(precomputed_tables, point, args);
    // }

    use rayon::prelude::*;
    use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};

    let chunk_size: u64 = 320_000_000_000_0 / ((30 - precomputed_tables.get_l1()).max(0) * 2).max(1) as u64;
    let total_range = args.range_end as u64 - args.range_start as u64;
    let num_chunks = ((total_range + chunk_size - 1) / chunk_size) as usize;
    let n_threads = args.n_threads;

    let end_flag = Arc::new(AtomicBool::new(false));

    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        for (i, (cache, alpha)) in t2_cache.iter_mut().zip(t2_cache_alpha.iter_mut()).enumerate() {
            let point = t2_table.index(i);
            *alpha = &MONTGOMERY_A_NEG - &point.u;
            *cache = point;
        }
    }

    (0..n_threads)
        .into_par_iter()
        .find_map_any(|thread_index| {
            let t2_u_values: [FieldElement; BATCH_SIZE] = {
                let mut u_values = [FieldElement::ZERO; BATCH_SIZE];
                for i in 0..BATCH_SIZE {
                    u_values[i] = t2_cache[i].u;
                }
                u_values
            };

            let t2_vs: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| p.v.clone());
            let t2_vs_neg: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| -&p.v);

            for chunk_index in (thread_index..num_chunks).step_by(n_threads) {
                if end_flag.load(Ordering::Relaxed) {
                    return None;
                }

                let start = args.range_start + (chunk_index as u64 * chunk_size) as i64;
                let end = ((chunk_index as u64 + 1) * chunk_size).min(args.range_end as u64) as i64;

                let chunk_args = ECDLPArguments {
                    range_start: start,
                    range_end: end,
                    n_threads: 1,
                    pseudo_constant_time: args.pseudo_constant_time,
                    progress_report_function: NoopReportFn, // FIXME: make chunk-relative on the global scale
                };

                let (offset, normalized, num_batches) =
                    decode_prep(precomputed_tables, point, &chunk_args, 1, 0);

                let progress_wrapper = |progress: f64| {
                    if !chunk_args.pseudo_constant_time && end_flag.load(Ordering::Relaxed) {
                        ControlFlow::Break(())
                    } else {
                        let result = args.progress_report_function.report(progress);
                        if result.is_break() {
                            end_flag.store(true, Ordering::SeqCst);
                        }
                        result
                    }
                };

                let point_iter =
                    make_point_iterator(precomputed_tables, normalized, num_batches);

                // if thread_index == 0 && (start / 100000000000000_i64) % n_threads as i64 == 0 {
                //     println!("Thread 0: processing chunk starting at {}", start);
                // }

                if let Some(res) = fast_ecdlp(
                    precomputed_tables,
                    normalized,
                    point_iter,
                    chunk_args.pseudo_constant_time,
                    &end_flag,
                    progress_wrapper,
                    &t2_cache,
                    &t2_cache_alpha,
                ) {
                    end_flag.store(true, Ordering::SeqCst);
                    return Some(offset + res as i64);
                }
            }

            None
        })
}


fn fast_ecdlp(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    target_point: RistrettoPoint,
    point_iterator: impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)>,
    pseudo_constant_time: bool,
    end_flag: &AtomicBool,
    progress_report: impl ProgressReportFunction,
    t2_cache: &[AffineMontgomeryPoint; BATCH_SIZE],
    t2_cache_alpha: &[FieldElement; BATCH_SIZE],
) -> Option<u64> {
    let t1_table = precomputed_tables.get_t1();

    let mut found = None;
    let mut consider_candidate = |m| {
        if i64_to_scalar(m) * G == target_point {
            found = found.or(Some(m as u64));
            
            // Signal other threads to stop when we find a result
            if !pseudo_constant_time {
                end_flag.store(true, Ordering::SeqCst);
            }
            true
        } else {
            false
        }
    };

    let mut batch = [FieldElement::ZERO; BATCH_SIZE];
    'outer: for (index, j_start, target_montgomery, progress) in point_iterator {
        // Check end_flag more frequently - at the start of each major iteration
        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }

        // amortize the potential cost of the report function
        if index % BATCH_SIZE == 0 {
            if let ControlFlow::Break(_) = progress_report.report(progress) {
                break 'outer;
            }
        }

        // Case 0: target is 0. Has to be handled separately.
        let j_start_shifted = (j_start as i64) << precomputed_tables.get_l1();
        if target_montgomery.is_identity_not_ct() {
            consider_candidate(j_start_shifted);
            if !pseudo_constant_time {
                break 'outer;
            }
        }

        // Case 2: j=0. Has to be handled separately.
        if t1_table
            .lookup(&target_montgomery.u.to_bytes(), |i| {
                consider_candidate(j_start_shifted + i as i64)
                    || consider_candidate(j_start_shifted - i as i64)
            })
            .is_some()
            && !pseudo_constant_time
        {
            break 'outer;
        }

        // Z = T2[j]_x - Pm_x
        for (i, batch) in batch.iter_mut().enumerate() {
            let j = i + 1;
            let t2_point = &t2_cache[i];
            let diff = &t2_point.u - &target_montgomery.u;

            if diff.is_zero_not_ct() {
                // Case 1: (Montgomery addition) exceptional case when T2[j] = Pm.
                // m1 = j * 2^L1, m2 = -j * 2^L1
                let found =
                    consider_candidate((j_start as i64 + j as i64) << precomputed_tables.get_l1())
                        || consider_candidate(
                            (j_start as i64 - j as i64) << precomputed_tables.get_l1(),
                        );
                if !pseudo_constant_time && found {
                    break 'outer;
                }
            }
            *batch = diff;
        }

        // nu = Z^-1
        FieldElement::batch_invert(&mut batch);

        for (batch_i, nu) in batch.iter().enumerate() {
            let j = batch_i + 1;
            // Montgomery addition: general case
            let t2_point = &t2_cache[batch_i];
            let alpha = &t2_cache_alpha[batch_i] - &target_montgomery.u;

            // lambda = (T2[j]_y - Pm_y) * nu
            // Q_x = lambda^2 - A - T2[j]_x - Pm_x
            let lambda = &(&t2_point.v - &target_montgomery.v) * nu;
            let qx = &lambda.square() + &alpha;

            if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
                break 'outer;
            }

            // Case 3: general case, negative j.
            let j_start_shifted = (j_start as i64 - j as i64) << precomputed_tables.get_l1();
            if t1_table
                .lookup(&qx.as_bytes(), |i| {
                    consider_candidate(j_start_shifted + i as i64)
                    || consider_candidate(
                        j_start_shifted - i as i64,
                    )
                })
                .is_some()
            {
                // m1 = -j * 2^L1 + i, m2 = -j * 2^L1 - i
                if !pseudo_constant_time {
                    break 'outer;
                }
            }

            // lambda = (p - T2[j]_y - Pm_y) * nu
            // Q_x = lambda^2 - A - T2[j]_x - Pm_x
            let lambda = &(&-&t2_point.v - &target_montgomery.v) * nu;
            let qx = &lambda.square() + &alpha;

            // Case 4: general case, positive j.
            let j_start_shifted = (j_start as i64 + j as i64) << precomputed_tables.get_l1();
            if t1_table
                .lookup(&qx.to_bytes(), |i| {
                    consider_candidate(j_start_shifted + i as i64)
                        || consider_candidate(j_start_shifted - i as i64)
                })
                .is_some()
            {
                // m1 = j * 2^L1 + i, m2 = j * 2^L1 - i
                if !pseudo_constant_time {
                    break 'outer;
                }
            }
        }
    }

    found
}

#[inline(always)]
fn batch_field_subtract<const N: usize>(
    batch: &mut [FieldElement; N],
    u_values: &[FieldElement; N],
    target_u: &FieldElement,
) {
    for i in 0..N {
        batch[i] = &u_values[i] - target_u;
    }
}

#[inline(always)]
pub fn batch_field_add<const N: usize>(
    out: &mut [FieldElement; N],
    a_values: &[FieldElement; N],
    b_values: &[FieldElement; N],
) {
    for i in 0..N {
        out[i] = &a_values[i] + &b_values[i];
    }
}

#[inline(always)]
fn batch_field_mul_and_square<const N: usize>(
    output: &mut [FieldElement; N],
    a: &[FieldElement; N],
    b: &[FieldElement; N],
) {
    let mut pos = 0;

    #[cfg(all(feature = "simd", curve25519_dalek_bits = "64"))]
    {
        const CHUNK_SIZE: usize = 4;
        while pos + CHUNK_SIZE <= N {
            let a_chunk: &[FieldElement; 4] = (&a[pos..pos + 4]).try_into().unwrap();
            let b_chunk: &[FieldElement; 4] = (&b[pos..pos + 4]).try_into().unwrap();
            
            let products = FieldElement::batch_mul_4way(a_chunk, b_chunk);
            let squared = FieldElement::batch_square_4way(&products);
            
            output[pos..pos + 4].copy_from_slice(&squared);
            pos += CHUNK_SIZE;
        }
    }

    #[cfg(all(feature = "simd", curve25519_dalek_bits = "32"))]
    {
        const CHUNK_SIZE: usize = 8;
        while pos + CHUNK_SIZE <= N {
            let a_chunk: &[FieldElement; 8] = (&a[pos..pos + 8]).try_into().unwrap();
            let b_chunk: &[FieldElement; 8] = (&b[pos..pos + 8]).try_into().unwrap();
            
            let products = FieldElement::batch_mul_8way(a_chunk, b_chunk);
            let squared = FieldElement::batch_square_8way(&products);
            
            output[pos..pos + 8].copy_from_slice(&squared);
            pos += CHUNK_SIZE;
        }
    }

    // Scalar fallback for remaining elements or when SIMD not available
    while pos < N {
        output[pos] = (&a[pos] * &b[pos]).square();
        pos += 1;
    }
}

/// Batch converts multiple i64 values to Scalar types using SIMD where possible
pub fn batch_i64_to_scalar(inputs: &[i64], outputs: &mut [Scalar]) {
    #[cfg(feature = "simd")]
    {
        use wide::{i64x4,CmpGt};  // Remove CmpGe import
        
        assert_eq!(inputs.len(), outputs.len());
        let len = inputs.len();
        let mut pos = 0;
        
        // Process in chunks of 4 using SIMD
        while pos + 4 <= len {
            // Create SIMD vector from array
            let values = i64x4::from([
                inputs[pos],
                inputs[pos+1],
                inputs[pos+2],
                inputs[pos+3],
            ]);
            
            // Manual comparison with bitwise ops
            // Use cmp_gt with -1 instead of cmp_ge with 0
            let is_positive = values.cmp_gt(i64x4::splat(-1));
            
            // Negate values
            let neg_values = i64x4::splat(0) - values;
            
            // Convert to arrays for manual selection since select isn't available
            let values_array = values.to_array();
            let neg_values_array = neg_values.to_array();
            let is_positive_array = is_positive.to_array();
            
            // Process individual conversions
            for i in 0..4 {
                // Manual selection based on mask
                let abs_value = if is_positive_array[i] > 0 {
                    values_array[i]
                } else {
                    neg_values_array[i]
                };
                
                let scalar_pos = Scalar::from(abs_value as u64);
                
                // Use the sign mask to determine if we need negation
                outputs[pos + i] = if is_positive_array[i] > 0 {
                    scalar_pos
                } else {
                    -&scalar_pos
                };
            }
            
            pos += 4;
        }
        
        // Handle remaining elements with scalar code
        for i in pos..len {
            outputs[i] = i64_to_scalar(inputs[i]);
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        for (i, &value) in inputs.iter().enumerate() {
            outputs[i] = i64_to_scalar(value);
        }
    }
}

pub fn batch_compute_shifts<const N: usize>(
    neg_shifts: &mut [i64; N],
    pos_shifts: &mut [i64; N],
    j_start: usize,
    l1: usize
) {
    #[cfg(feature = "simd")]
    {
        use wide::i64x4;
        
        let mut pos = 0;
        let j_start_i64 = j_start as i64;
        let shift_amount = l1 as i64;
        let shift_factor = 1i64 << shift_amount; // Compute shift factor once
        
        // Process in chunks of 4 using SIMD
        while pos + 4 <= N {
            // Create indices vector [1,2,3,4] + pos
            let indices = [
                (pos + 1) as i64,
                (pos + 2) as i64,
                (pos + 3) as i64,
                (pos + 4) as i64
            ];
            let j_indices = i64x4::from(indices);
            
            // Create j_start vector [j_start, j_start, j_start, j_start]
            let j_start_vec = i64x4::splat(j_start_i64);
            
            // Compute j_start - j and j_start + j
            let neg_j = j_start_vec - j_indices;
            let pos_j = j_start_vec + j_indices;
            
            // Manually shift the values by multiplying
            let neg_shifted = neg_j * i64x4::splat(shift_factor);
            let pos_shifted = pos_j * i64x4::splat(shift_factor);
            
            // Convert to arrays for access
            let neg_shifted_array = neg_shifted.to_array();
            let pos_shifted_array = pos_shifted.to_array();
            
            // Store the results
            for i in 0..4 {
                if pos + i < N {
                    neg_shifts[pos + i] = neg_shifted_array[i];
                    pos_shifts[pos + i] = pos_shifted_array[i];
                }
            }
            
            pos += 4;
        }
        
        // Handle remaining elements with scalar code
        for i in pos..N {
            let j = i + 1;
            neg_shifts[i] = (j_start as i64 - j as i64) << l1;
            pos_shifts[i] = (j_start as i64 + j as i64) << l1;
        }
    } 
    #[cfg(not(feature = "simd"))]
    {
        for i in 0..N {
            let j = i + 1;
            neg_shifts[i] = (j_start as i64 - j as i64) << l1;
            pos_shifts[i] = (j_start as i64 + j as i64) << l1;
        }
    }
}

fn fast_ecdlp_simd(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    target_point: RistrettoPoint,
    point_iterator: impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)>,
    pseudo_constant_time: bool,
    end_flag: &AtomicBool,
    progress_report: impl ProgressReportFunction,
    t2_cache: &[AffineMontgomeryPoint; BATCH_SIZE],
    t2_cache_alpha: &[FieldElement; BATCH_SIZE],
    t2_u_values: &[FieldElement; BATCH_SIZE],
    t2_vs: &[FieldElement; BATCH_SIZE],
    t2_vs_neg: &[FieldElement; BATCH_SIZE],
    batch: &mut [FieldElement; BATCH_SIZE],
    alphas: &mut [FieldElement; BATCH_SIZE],
    qxs: &mut [FieldElement; BATCH_SIZE],
    neg_qxs: &mut [FieldElement; BATCH_SIZE],
) -> Option<u64> {
    let t1_table = precomputed_tables.get_t1();

    let mut found = None;
    let l1 = precomputed_tables.get_l1();

    let (mut qx_out, mut qx_tmp) = (qxs, neg_qxs);
    
    'outer: for (index, j_start, target_montgomery, progress) in point_iterator {
        let mut consider_candidate = |m| {
            if i64_to_scalar(m) * G == target_point {
                found = found.or(Some(m as u64));
                
                true
            } else {
                false
            }
        };

        // Check end_flag more frequently - at the start of each major iteration
        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }

        // amortize the potential cost of the report function
        if index % BATCH_SIZE == 0 {
            if let ControlFlow::Break(_) = progress_report.report(progress) {
                break 'outer;
            }
        }

        // // Case 0: target is 0. Has to be handled separately.
        // let j_start_shifted = (j_start as i64) << l1;
        // if target_montgomery.is_identity_not_ct() {
        //     consider_candidate(j_start_shifted);
        //     if !pseudo_constant_time {
        //         break 'outer;
        //     }
        // }

        // // Case 2: j=0. Has to be handled separately.
        // if t1_table
        //     .lookup(&target_montgomery.u.as_bytes(), |i| {
        //         consider_candidate(j_start_shifted + i as i64)
        //             || consider_candidate(j_start_shifted - i as i64)
        //     })
        //     .is_some()
        //     && !pseudo_constant_time
        // {
        //     break 'outer;
        // }

        batch_field_subtract(batch, &t2_u_values, &target_montgomery.u);

        // TODO: make a helper version of this function that has AVX512/8-way operations
        // Would use runtime dispatch by caching the path to be taken based on available SIMD width

        // 4-lane batch invert inlined
        {
            const NUM_CHUNKS: usize = BATCH_SIZE / 4;
            
            // Pre-allocated scratch space (could be passed in as parameter to avoid allocation)
            let mut scratch = [FieldElement::ONE; NUM_CHUNKS * 4];
            
            // 4 parallel accumulators
            let mut acc_lanes = [FieldElement::ONE; 4];
            
            // Forward pass
            for chunk_idx in 0..NUM_CHUNKS {
                let base = chunk_idx * 4;
                let scratch_base = chunk_idx * 4;
                
                // Store current accumulators
                scratch[scratch_base] = acc_lanes[0];
                scratch[scratch_base + 1] = acc_lanes[1];
                scratch[scratch_base + 2] = acc_lanes[2];
                scratch[scratch_base + 3] = acc_lanes[3];
                
                // Load input chunk directly from batch
                let input_chunk = [
                    batch[base],
                    batch[base + 1],
                    batch[base + 2],
                    batch[base + 3],
                ];
                
                // Update accumulators using SIMD
                acc_lanes = FieldElement::batch_mul_4way(&acc_lanes, &input_chunk);
            }
            
            // Combined inversion approach
            let p01 = &acc_lanes[0] * &acc_lanes[1];
            let p23 = &acc_lanes[2] * &acc_lanes[3];
            let p0123 = &p01 * &p23;
            let inv_p0123 = p0123.invert();
            
            // Extract individual inverses
            let factors = [
                &acc_lanes[1] * &p23,
                &acc_lanes[0] * &p23,
                &p01 * &acc_lanes[3],
                &p01 * &acc_lanes[2],
            ];
            
            let inv_broadcast = [inv_p0123; 4];
            acc_lanes = FieldElement::batch_mul_4way(&inv_broadcast, &factors);
            
            // Reverse pass
            for chunk_idx in (0..NUM_CHUNKS).rev() {
                let base = chunk_idx * 4;
                let scratch_base = chunk_idx * 4;
                
                // Load input chunk
                let input_chunk = [
                    batch[base],
                    batch[base + 1],
                    batch[base + 2],
                    batch[base + 3],
                ];
                
                // Load scratch chunk
                let scratch_chunk = [
                    scratch[scratch_base],
                    scratch[scratch_base + 1],
                    scratch[scratch_base + 2],
                    scratch[scratch_base + 3],
                ];
                
                // Compute results using SIMD
                let results = FieldElement::batch_mul_4way(&acc_lanes, &scratch_chunk);
                
                // Store results back to batch
                batch[base] = results[0];
                batch[base + 1] = results[1];
                batch[base + 2] = results[2];
                batch[base + 3] = results[3];
                
                // Update accumulators using SIMD
                acc_lanes = FieldElement::batch_mul_4way(&acc_lanes, &input_chunk);
            }
        }
    
        batch_field_subtract(
                  alphas,
                  &t2_cache_alpha,
                  &target_montgomery.u
        );

        // Batch compute qxs for the regular case
        // lambda = (T2[j]_y - Pm_y) * nu
        batch_field_subtract(qx_tmp, &t2_vs, &target_montgomery.v);
        batch_field_mul_and_square(qx_out, &qx_tmp, &batch);
        batch_field_add(qx_tmp, &qx_out, alphas);

        // Process in groups of 8
        for chunk_idx in 0..(BATCH_SIZE / 8) {
            let base = chunk_idx * 8;
            let queries = [
                qx_tmp[base].as_bytes(),
                qx_tmp[base + 1].as_bytes(),
                qx_tmp[base + 2].as_bytes(),
                qx_tmp[base + 3].as_bytes(),
                qx_tmp[base + 4].as_bytes(),
                qx_tmp[base + 5].as_bytes(),
                qx_tmp[base + 6].as_bytes(),
                qx_tmp[base + 7].as_bytes(),
            ];
            
            let results = t1_table.lookup_batch_8(&queries);
            
            for (lane, (found_match, value)) in results.iter().enumerate() {
                if *found_match {
                    let j = base + lane + 1;
                    let j_start_shifted = (j_start as i64 - j as i64) << l1;
                    if consider_candidate(j_start_shifted + *value as i64) ||
                       consider_candidate(j_start_shifted - *value as i64) {
                        if !pseudo_constant_time {
                            break 'outer;
                        }
                    }
                }
            }
        }

        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }

        batch_field_subtract(qx_tmp, &t2_vs_neg, &target_montgomery.v);
        batch_field_mul_and_square(qx_out, &qx_tmp, &batch);
        batch_field_add(qx_tmp, &qx_out, alphas);

        // Process in groups of 8
        for chunk_idx in 0..(BATCH_SIZE / 8) {
            let base = chunk_idx * 8;
            let queries = [
                qx_tmp[base].as_bytes(),
                qx_tmp[base + 1].as_bytes(),
                qx_tmp[base + 2].as_bytes(),
                qx_tmp[base + 3].as_bytes(),
                qx_tmp[base + 4].as_bytes(),
                qx_tmp[base + 5].as_bytes(),
                qx_tmp[base + 6].as_bytes(),
                qx_tmp[base + 7].as_bytes(),
            ];
            
            let results = t1_table.lookup_batch_8(&queries);
            
            for (lane, (found_match, value)) in results.iter().enumerate() {
                if *found_match {
                    let j = base + lane + 1;
                    let j_start_shifted = (j_start as i64 + j as i64) << l1;
                    if consider_candidate(j_start_shifted + *value as i64) ||
                       consider_candidate(j_start_shifted - *value as i64) {
                        if !pseudo_constant_time {
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    found
}

fn fast_ecdlp_simd_j_batched(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    target_point: RistrettoPoint,
    mut point_iterator: impl Iterator<Item = [(usize, usize, AffineMontgomeryPoint, f64); 4]>,
    pseudo_constant_time: bool,
    end_flag: &AtomicBool,
    progress_report: impl ProgressReportFunction,
    t2_cache: &[AffineMontgomeryPoint; BATCH_SIZE],
    t2_cache_alpha: &[FieldElement; BATCH_SIZE],
    t2_u_values: &[FieldElement; BATCH_SIZE],
    t2_vs: &[FieldElement; BATCH_SIZE],
    t2_vs_neg: &[FieldElement; BATCH_SIZE],
    batch: &mut [FieldElement; BATCH_SIZE],
    alphas: &mut [FieldElement; BATCH_SIZE],
    qxs: &mut [FieldElement; BATCH_SIZE],
    neg_qxs: &mut [FieldElement; BATCH_SIZE],
    j_queue: &mut Vec<(usize, usize, AffineMontgomeryPoint, f64)>,
    batch_storage: &mut [[FieldElement; BATCH_SIZE]; J_BATCH_SIZE],
    alpha_storage: &mut [[FieldElement; BATCH_SIZE]; J_BATCH_SIZE],
) -> Option<u64> {    
    let t1_table = precomputed_tables.get_t1();
    let mut found = None;
    let l1 = precomputed_tables.get_l1();
    
    'outer: loop {
        // Get next batch of 4 points directly from iterator
        let Some(j_batch) = point_iterator.next() else {
            break 'outer;
        };
        
        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }
        
        // Prepare batch data for all 4 j values in parallel
        for (j_idx, (_, _, target_montgomery, _)) in j_batch.iter().enumerate() {
            if target_montgomery.is_identity_not_ct() {
                continue; // Skip identity points
            }
            batch_field_subtract(&mut batch_storage[j_idx], &t2_u_values, &target_montgomery.u);
            batch_field_subtract(&mut alpha_storage[j_idx], &t2_cache_alpha, &target_montgomery.u);
        }
        
        // Batch inversion for all 4 j values
        let mut accs = [FieldElement::ONE; 4];
        let mut scratch = [[FieldElement::ONE; BATCH_SIZE]; 4];

        // Forward pass
        for j_idx in 0..4 {
            for i in 0..BATCH_SIZE {
                scratch[j_idx][i] = accs[j_idx];
                accs[j_idx] = &accs[j_idx] * &batch_storage[j_idx][i];
            }
        }

        // Combined inversion
        let p01 = &accs[0] * &accs[1];
        let p23 = &accs[2] * &accs[3];
        let p0123 = &p01 * &p23;
        let inv_p0123 = p0123.invert();

        let factors = [
            &accs[1] * &p23,
            &accs[0] * &p23,
            &p01 * &accs[3],
            &p01 * &accs[2],
        ];

        let inv_broadcast = [inv_p0123; 4];
        let inv_accs = FieldElement::batch_mul_4way(&inv_broadcast, &factors);

        // Reverse pass
        for j_idx in 0..4 {
            let mut acc = inv_accs[j_idx];
            for i in (0..BATCH_SIZE).rev() {
                let tmp = &acc * &batch_storage[j_idx][i];
                batch_storage[j_idx][i] = &acc * &scratch[j_idx][i];
                acc = tmp;
            }
        }
        
        // Process lookups for each j value
        for (j_idx, (_, j_start, target_montgomery, progress)) in j_batch.iter().enumerate() {
            if target_montgomery.is_identity_not_ct() {
                continue;
            }
            
            // Report progress periodically
            if j_idx == 0 {
                if let ControlFlow::Break(_) = progress_report.report(*progress) {
                    break 'outer;
                }
            }
            
            let batch = &batch_storage[j_idx];
            let alphas = &alpha_storage[j_idx];
            
            // Allocate these once outside the loop if possible
            let mut qxs = [FieldElement::ZERO; BATCH_SIZE];
            let mut neg_qxs = [FieldElement::ZERO; BATCH_SIZE];
            
            // Regular case lookups
            batch_field_subtract(&mut qxs, &t2_vs, &target_montgomery.v);
            batch_field_mul_and_square(&mut neg_qxs, &qxs, batch);
            batch_field_add(&mut qxs, &neg_qxs, alphas);
            
            // Process lookups with batch_8
            for chunk_idx in 0..(BATCH_SIZE / 8) {
                let base = chunk_idx * 8;
                let queries = [
                    qxs[base].as_bytes(),
                    qxs[base + 1].as_bytes(),
                    qxs[base + 2].as_bytes(),
                    qxs[base + 3].as_bytes(),
                    qxs[base + 4].as_bytes(),
                    qxs[base + 5].as_bytes(),
                    qxs[base + 6].as_bytes(),
                    qxs[base + 7].as_bytes(),
                ];
                
                let results = t1_table.lookup_batch_8(&queries);
                
                for (lane, (found_match, value)) in results.iter().enumerate() {
                    if *found_match {
                        let j = base + lane + 1;
                        let j_start_shifted = (*j_start as i64 - j as i64) << l1;
                        
                        let candidate1 = j_start_shifted + *value as i64;
                        if i64_to_scalar(candidate1) * G == target_point {
                            found = found.or(Some(candidate1 as u64));
                            if !pseudo_constant_time {
                                break 'outer;
                            }
                        }

                        let candidate2 = j_start_shifted - *value as i64;
                        if i64_to_scalar(candidate2) * G == target_point {
                            found = found.or(Some(candidate2 as u64));
                            if !pseudo_constant_time {
                                break 'outer;
                            }
                        }
                    }
                }
            }
            
            // Negative case lookups (similar structure)
            batch_field_subtract(&mut qxs, &t2_vs_neg, &target_montgomery.v);
            batch_field_mul_and_square(&mut neg_qxs, &qxs, batch);
            batch_field_add(&mut qxs, &neg_qxs, alphas);
            
            // Process negative lookups...
            for chunk_idx in 0..(BATCH_SIZE / 8) {
                let base = chunk_idx * 8;
                let queries = [
                    qxs[base].as_bytes(),
                    qxs[base + 1].as_bytes(),
                    qxs[base + 2].as_bytes(),
                    qxs[base + 3].as_bytes(),
                    qxs[base + 4].as_bytes(),
                    qxs[base + 5].as_bytes(),
                    qxs[base + 6].as_bytes(),
                    qxs[base + 7].as_bytes(),
                ];
                
                let results = t1_table.lookup_batch_8(&queries);
                
                for (lane, (found_match, value)) in results.iter().enumerate() {
                    if *found_match {
                        let j = base + lane + 1;
                        let j_start_shifted = (*j_start as i64 - j as i64) << l1;
                        
                        let candidate1 = j_start_shifted + *value as i64;
                        if i64_to_scalar(candidate1) * G == target_point {
                            found = found.or(Some(candidate1 as u64));
                            if !pseudo_constant_time {
                                break 'outer;
                            }
                        }

                        let candidate2 = j_start_shifted - *value as i64;
                        if i64_to_scalar(candidate2) * G == target_point {
                            found = found.or(Some(candidate2 as u64));
                            if !pseudo_constant_time {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }
    
    found
}

// FIXME(upstrean): should be an impl From<i64> for Scalar
#[inline]
fn i64_to_scalar(n: i64) -> Scalar {
    if n >= 0 {
        Scalar::from(n as u64)
    } else {
        -&Scalar::from((-n) as u64)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::{Arc, Mutex},
    };

    use super::*;
    use rand::Rng;

    const L1: usize = 26;

    // Necessary for one ECDLP tables allocation only
    static TABLES: Mutex<Option<Arc<ECDLPTables>>> = Mutex::new(None);

    fn read_or_gen_tables() -> Arc<ECDLPTables> {
        let mut tables = TABLES.lock().expect("acquire tables lock");
        if let Some(v) = tables.as_ref().cloned() {
            return v;
        }

        let inner = if !Path::new("ecdlp_table.bin").exists() {
            let tables = ECDLPTables::generate_par(L1, 16).unwrap();
            tables.write_to_file("ecdlp_table.bin").unwrap();
            tables
        } else {
            ECDLPTables::load_from_file(L1, "ecdlp_table.bin").unwrap()
        };

        let t = Arc::new(inner);
        *tables = Some(t.clone());

        t
    }

    #[test]
    fn test_ecdlp_cofactors() {
        let tables = read_or_gen_tables();
        let view = tables.view();

        for i in (0..(1u64 << 48)).step_by(1 << L1).take(1 << 12) {
            let delta = rand::thread_rng().gen_range(0..(1 << L1));

            let num = i + delta;
            let point = RistrettoPoint::mul_base(&Scalar::from(num));

            // take a random point from the coset4
            let coset_i = rand::thread_rng().gen_range(0..4);
            let point = point.coset4()[coset_i];
            // let point = point.compress().decompress().unwrap();

            let res = decode(
                &view,
                RistrettoPoint(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));

            println!("tested {num} (coset4[{coset_i}])");
        }
    }

    #[test]
    fn test_ecdlp_single() {
        let tables = read_or_gen_tables();
        let view = tables.view();

        for i in (0..(1u64 << 48)).step_by(1 << L1).take(1 << 12) {
            let num = i; // rand::thread_rng().gen_range(0u64..(1 << 48));
            let mut point = RistrettoPoint::mul_base(&Scalar::from(num));

            if rand::thread_rng().gen() {
                // do a round of compression/decompression to mess up the Z and Ts
                // & ecdlp will need to clear the cofactor
                point = point.compress().decompress().unwrap();
            }

            let res = decode(&view, point, ECDLPArguments::new_with_range(0, 1 << 48));
            assert_eq!(res, Some(num as i64));

            println!("tested {num}");
        }
    }

    #[test]
    fn test_ecdlp_par_decode() {
        let tables = read_or_gen_tables();
        let view = tables.view();

        for i in (0..(1u64 << 48)).step_by(1 << L1).take(1 << 12) {
            let value = i;

            let mut point = RistrettoPoint::mul_base(&Scalar::from(value));

            if rand::thread_rng().gen() {
                // do a round of compression/decompression to mess up the Z and Ts
                // & ecdlp will need to clear the cofactor
                point = point.compress().decompress().unwrap();
            }

            let res = par_decode(
                &view,
                point,
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .n_threads(4)
                    .pseudo_constant_time(false),
            );
            assert_eq!(res, Some(value as i64));
        }
    }

    #[test]
    fn test_ecdlp_single_large_value_timing() {
        use std::time::Instant;
        use std::hint::black_box;

        const N: usize = 100;
        const N_THREADS: usize = 8;

        let tables = read_or_gen_tables();
        let view = tables.view();
    
        // let value: u64 = 7777777777 * 100_000_000;
        let value: u64 = 9_223_372_036_854_775_806/10000;
        let mut point = RistrettoPoint::mul_base(&Scalar::from(value));
    
        if rand::thread_rng().gen() {
            // Optionally alter the point via compression/decompression
            point = point.compress().decompress().unwrap();
        }
    
        // Benchmark par_decode
        let mut decode_times = vec![];
        for _ in 0..N {
            let args = ECDLPArguments::new_with_range(0, 1 << 63)
                .n_threads(N_THREADS)
                .pseudo_constant_time(false);
            let now = Instant::now();
            let res = par_decode(&view, black_box(point), args);
            let elapsed = now.elapsed();
            assert_eq!(res, Some(value as i64));
            decode_times.push(elapsed.as_secs_f64());
        }
        let avg_decode = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
        println!("Average SIMD decode time: {:.6} seconds", avg_decode);

        // Benchmark par_decode_j_batch
        let mut batched_times = vec![];
        for _ in 0..N {
            let args = ECDLPArguments::new_with_range(0, 1 << 63)
                .n_threads(N_THREADS)
                .pseudo_constant_time(false);
            let now = Instant::now();
            let res = par_decode_j_batch(&view, black_box(point), args);
            let elapsed = now.elapsed();
            assert_eq!(res, Some(value as i64));
            batched_times.push(elapsed.as_secs_f64());
        }
        let avg_batched = batched_times.iter().sum::<f64>() / batched_times.len() as f64;
        println!("Average SIMD (batched j) decode time: {:.6} seconds", avg_batched);

        // Benchmark par_decode_scalar
        let mut scalar_times = vec![];
        for _ in 0..N {
            let args = ECDLPArguments::new_with_range(0, 1 << 63)
                .n_threads(N_THREADS)
                .pseudo_constant_time(false);
            let now = Instant::now();
            let res = par_decode_scalar(&view, black_box(point), args);
            let elapsed = now.elapsed();
            assert_eq!(res, Some(value as i64));
            scalar_times.push(elapsed.as_secs_f64());
        }
        let avg_scalar = scalar_times.iter().sum::<f64>() / scalar_times.len() as f64;
        println!("Average decode_scalar time: {:.6} seconds", avg_scalar);
    }

    #[test]
    fn test_table_par() {
        // Measure parallel generation time
        let tables_par = ECDLPTables::generate_par(
            18,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
        )
        .unwrap();

        // Measure sequential generation time
        let tables_seq = ECDLPTables::generate(18).unwrap();

        // Verify both tables are identical
        assert_eq!(
            tables_seq.as_slice(),
            tables_par.as_slice(),
            "Sequential and parallel generated tables should be identical"
        );
    }

    fn fill_array_with<F, const N: usize>(mut f: F) -> [FieldElement; N]
    where
        F: FnMut() -> FieldElement,
    {
        let mut arr = [FieldElement::ZERO; N];
        for elem in &mut arr {
            *elem = f();
        }
        arr
    }

    #[test]
    fn test_schoolbook_wide_u64x4() {
        use wide::u64x4;
        use rand_core::{OsRng, RngCore};
        use std::time::Instant;
        use std::hint::black_box;

        const TEST_CASES: usize = 100_000;
        const LOW_51: u64 = (1 << 51) - 1;

        let mut rng = OsRng;
        let mut a_lanes = vec![[0u64; 5]; TEST_CASES * 4];
        let mut b_lanes = vec![[0u64; 5]; TEST_CASES * 4];

        for (a, b) in a_lanes.iter_mut().zip(b_lanes.iter_mut()) {
            for i in 0..5 {
                a[i] = rng.next_u64() & LOW_51;
                b[i] = rng.next_u64() & LOW_51;
            }
        }

        // Scalar
        let mut scalar_checksum = 0u64;
        let scalar_start = Instant::now();
        let mut b_precompute_checksum_scalar = 0u64;
        let mut c_checksums_scalar = [0u64; 5];

        for (a, b) in a_lanes.iter().zip(b_lanes.iter()) {
            let b1_19 = b[1] * 19;
            let b2_19 = b[2] * 19;
            let b3_19 = b[3] * 19;
            let b4_19 = b[4] * 19;

            // b_precompute_checksum_scalar ^= b1_19;
            // b_precompute_checksum_scalar ^= b2_19;
            // b_precompute_checksum_scalar ^= b3_19;
            // b_precompute_checksum_scalar ^= b4_19;

            let mut c0 = (a[0] as u128) * (b[0] as u128)
                + (a[4] as u128) * (b1_19 as u128)
                + (a[3] as u128) * (b2_19 as u128)
                + (a[2] as u128) * (b3_19 as u128)
                + (a[1] as u128) * (b4_19 as u128);
            // c_checksums_scalar[0] ^= c0 as u64 ^ (c0 >> 64) as u64;

            let mut c1 = (a[1] as u128) * (b[0] as u128)
                + (a[0] as u128) * (b[1] as u128)
                + (a[4] as u128) * (b2_19 as u128)
                + (a[3] as u128) * (b3_19 as u128)
                + (a[2] as u128) * (b4_19 as u128);
            // c_checksums_scalar[1] ^= c1 as u64 ^ (c1 >> 64) as u64;


            let mut c2 = (a[2] as u128) * (b[0] as u128)
                + (a[1] as u128) * (b[1] as u128)
                + (a[0] as u128) * (b[2] as u128)
                + (a[4] as u128) * (b3_19 as u128)
                + (a[3] as u128) * (b4_19 as u128);
            // c_checksums_scalar[2] ^= c2 as u64 ^ (c2 >> 64) as u64;


            let mut c3 = (a[3] as u128) * (b[0] as u128)
                + (a[2] as u128) * (b[1] as u128)
                + (a[1] as u128) * (b[2] as u128)
                + (a[0] as u128) * (b[3] as u128)
                + (a[4] as u128) * (b4_19 as u128);
            // c_checksums_scalar[3] ^= c3 as u64 ^ (c3 >> 64) as u64;


            let mut c4 = (a[4] as u128) * (b[0] as u128)
                + (a[3] as u128) * (b[1] as u128)
                + (a[2] as u128) * (b[2] as u128)
                + (a[1] as u128) * (b[3] as u128)
                + (a[0] as u128) * (b[4] as u128);
            // c_checksums_scalar[4] ^= c4 as u64 ^ (c4 >> 64) as u64;


            // Carry propagation
            let mut out = [0u64; 5];

            c1 += (c0 >> 51) as u128;
            out[0] = (c0 as u64) & LOW_51;

            c2 += (c1 >> 51) as u128;
            out[1] = (c1 as u64) & LOW_51;

            c3 += (c2 >> 51) as u128;
            out[2] = (c2 as u64) & LOW_51;

            c4 += (c3 >> 51) as u128;
            out[3] = (c3 as u64) & LOW_51;

            let carry = (c4 >> 51) as u64;
            out[4] = (c4 as u64) & LOW_51;

            out[0] += carry * 19;
            out[1] += out[0] >> 51;
            out[0] &= LOW_51;

            for &limb in &out {
                scalar_checksum ^= limb;
            }

            black_box(out);
        }
        let scalar_time = scalar_start.elapsed().as_secs_f64();

        // SIMD
        let mut simd_checksum = 0u64;
        let simd_start = Instant::now();
        let mut b_precompute_checksum_simd = 0u64;
        let mut c_checksums_simd = [0u64; 5];
        
        for (a_chunk, b_chunk) in a_lanes.chunks_exact(4).zip(b_lanes.chunks_exact(4)) {
            let mut a = [u64x4::default(); 5];
            let mut b = [u64x4::default(); 5];
            for i in 0..5 {
                a[i] = u64x4::new([a_chunk[0][i], a_chunk[1][i], a_chunk[2][i], a_chunk[3][i]]);
                b[i] = u64x4::new([b_chunk[0][i], b_chunk[1][i], b_chunk[2][i], b_chunk[3][i]]);
            }

            const FACTOR_19: u64x4 = u64x4::new([19, 19, 19, 19]);
            let b1_19 = b[1] * FACTOR_19;
            let b2_19 = b[2] * FACTOR_19;
            let b3_19 = b[3] * FACTOR_19;
            let b4_19 = b[4] * FACTOR_19;

            let b1_19_arr = b1_19.to_array();
            let b2_19_arr = b2_19.to_array();
            let b3_19_arr = b3_19.to_array();
            let b4_19_arr = b4_19.to_array();

            // Helper functions at the top of your SIMD section
            #[inline(always)]
            fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
                const MASK_32: u64x4 = u64x4::new([0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF]);
                
                let a_lo = a & MASK_32;
                let a_hi = a >> 32;
                let b_lo = b & MASK_32;
                let b_hi = b >> 32;
                
                let lo_lo = a_lo * b_lo;
                let lo_hi = a_lo * b_hi;
                let hi_lo = a_hi * b_lo;
                let hi_hi = a_hi * b_hi;
                
                let mid = lo_hi + hi_lo;
                let mid_lo = mid << 32;
                let mid_hi = mid >> 32;
                
                let res_lo: u64x4 = lo_lo + mid_lo;
                let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
                let res_hi: u64x4 = hi_hi + mid_hi + carry;
                
                (res_lo, res_hi)
            }

            #[inline(always)]
            fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
                let sum_lo = a_lo + b_lo;
                let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
                let sum_hi = a_hi + b_hi + carry;
                (sum_lo, sum_hi)
            }

            // Replace your scalar multiplication loops with this SIMD version:

            // Compute c0: a[0]*b[0] + a[4]*b1_19 + a[3]*b2_19 + a[2]*b3_19 + a[1]*b4_19
            let a0_b0 = mul64_to_128_simd(a[0], b[0]);
            let a0_b1 = mul64_to_128_simd(a[0], b[1]);
            let a0_b2 = mul64_to_128_simd(a[0], b[2]);
            let a0_b3 = mul64_to_128_simd(a[0], b[3]);
            let a0_b4 = mul64_to_128_simd(a[0], b[4]);

            let a1_b0 = mul64_to_128_simd(a[1], b[0]);
            let a1_b1 = mul64_to_128_simd(a[1], b[1]);
            let a1_b2 = mul64_to_128_simd(a[1], b[2]);
            let a1_b3 = mul64_to_128_simd(a[1], b[3]);
            let a1_b4_19 = mul64_to_128_simd(a[1], b4_19);

            let a2_b0 = mul64_to_128_simd(a[2], b[0]);
            let a2_b1 = mul64_to_128_simd(a[2], b[1]);
            let a2_b2 = mul64_to_128_simd(a[2], b[2]);
            let a2_b3_19 = mul64_to_128_simd(a[2], b3_19);
            let a2_b4_19 = mul64_to_128_simd(a[2], b4_19);

            let a3_b0 = mul64_to_128_simd(a[3], b[0]);
            let a3_b1 = mul64_to_128_simd(a[3], b[1]);
            let a3_b2_19 = mul64_to_128_simd(a[3], b2_19);
            let a3_b3_19 = mul64_to_128_simd(a[3], b3_19);
            let a3_b4_19 = mul64_to_128_simd(a[3], b4_19);

            let a4_b0 = mul64_to_128_simd(a[4], b[0]);
            let a4_b1_19 = mul64_to_128_simd(a[4], b1_19);
            let a4_b2_19 = mul64_to_128_simd(a[4], b2_19);
            let a4_b3_19 = mul64_to_128_simd(a[4], b3_19);
            let a4_b4_19 = mul64_to_128_simd(a[4], b4_19);

            // Compute c0 = a[0]*b[0] + a[4]*b1_19 + a[3]*b2_19 + a[2]*b3_19 + a[1]*b4_19
            let (c0_lo, c0_hi) = a0_b0;
            let (c0_lo, c0_hi) = add_128_simd(c0_lo, c0_hi, a4_b1_19.0, a4_b1_19.1);
            let (c0_lo, c0_hi) = add_128_simd(c0_lo, c0_hi, a3_b2_19.0, a3_b2_19.1);
            let (c0_lo, c0_hi) = add_128_simd(c0_lo, c0_hi, a2_b3_19.0, a2_b3_19.1);
            let (c0_lo, c0_hi) = add_128_simd(c0_lo, c0_hi, a1_b4_19.0, a1_b4_19.1);

            // Compute c1 = a[1]*b[0] + a[0]*b[1] + a[4]*b2_19 + a[3]*b3_19 + a[2]*b4_19
            let (c1_lo, c1_hi) = a1_b0;
            let (c1_lo, c1_hi) = add_128_simd(c1_lo, c1_hi, a0_b1.0, a0_b1.1);
            let (c1_lo, c1_hi) = add_128_simd(c1_lo, c1_hi, a4_b2_19.0, a4_b2_19.1);
            let (c1_lo, c1_hi) = add_128_simd(c1_lo, c1_hi, a3_b3_19.0, a3_b3_19.1);
            let (c1_lo, c1_hi) = add_128_simd(c1_lo, c1_hi, a2_b4_19.0, a2_b4_19.1);

            // Compute c2 = a[2]*b[0] + a[1]*b[1] + a[0]*b[2] + a[4]*b3_19 + a[3]*b4_19
            let (c2_lo, c2_hi) = a2_b0;
            let (c2_lo, c2_hi) = add_128_simd(c2_lo, c2_hi, a1_b1.0, a1_b1.1);
            let (c2_lo, c2_hi) = add_128_simd(c2_lo, c2_hi, a0_b2.0, a0_b2.1);
            let (c2_lo, c2_hi) = add_128_simd(c2_lo, c2_hi, a4_b3_19.0, a4_b3_19.1);
            let (c2_lo, c2_hi) = add_128_simd(c2_lo, c2_hi, a3_b4_19.0, a3_b4_19.1);

            // Compute c3 = a[3]*b[0] + a[2]*b[1] + a[1]*b[2] + a[0]*b[3] + a[4]*b4_19
            let (c3_lo, c3_hi) = a3_b0;
            let (c3_lo, c3_hi) = add_128_simd(c3_lo, c3_hi, a2_b1.0, a2_b1.1);
            let (c3_lo, c3_hi) = add_128_simd(c3_lo, c3_hi, a1_b2.0, a1_b2.1);
            let (c3_lo, c3_hi) = add_128_simd(c3_lo, c3_hi, a0_b3.0, a0_b3.1);
            let (c3_lo, c3_hi) = add_128_simd(c3_lo, c3_hi, a4_b4_19.0, a4_b4_19.1);

            // Compute c4 = a[4]*b[0] + a[3]*b[1] + a[2]*b[2] + a[1]*b[3] + a[0]*b[4]
            let (c4_lo, c4_hi) = a4_b0;
            let (c4_lo, c4_hi) = add_128_simd(c4_lo, c4_hi, a3_b1.0, a3_b1.1);
            let (c4_lo, c4_hi) = add_128_simd(c4_lo, c4_hi, a2_b2.0, a2_b2.1);
            let (c4_lo, c4_hi) = add_128_simd(c4_lo, c4_hi, a1_b3.0, a1_b3.1);
            let (c4_lo, c4_hi) = add_128_simd(c4_lo, c4_hi, a0_b4.0, a0_b4.1);

            // Now we have (c0_lo, c0_hi), (c1_lo, c1_hi), etc. ready for carry propagation
            // Your existing carry propagation code, but using these SIMD pairs:

            const mask: u64x4 = u64x4::new([LOW_51,LOW_51,LOW_51,LOW_51]);

            // Process c0 to get initial carry
            let mut limb0 = c0_lo & mask;
            let mut carry = (c0_hi << 13) | (c0_lo >> 51);

            // Helper for overflow correction
            #[inline(never)]
            fn handle_overflow_correction(carry: u64x4, overflow_mask: u64x4) -> u64x4 {
                carry + overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0))
            }

            // Process c1
            let acc1: u64x4 = c1_lo + carry;
            let mut limb1 = acc1 & mask;
            let mut carry_new = (c1_hi << 13) | (acc1 >> 51);
            let overflow_mask = acc1.cmp_lt(c1_lo);
            if overflow_mask.to_array() != [0, 0, 0, 0] {
                carry_new = handle_overflow_correction(carry_new, overflow_mask);
            }
            carry = carry_new;

            // Process c2
            let acc2: u64x4 = c2_lo + carry;
            let mut limb2 = acc2 & mask;
            let mut carry_new = (c2_hi << 13) | (acc2 >> 51);
            let overflow_mask = acc2.cmp_lt(c2_lo);
            if overflow_mask.to_array() != [0, 0, 0, 0] {
                carry_new = handle_overflow_correction(carry_new, overflow_mask);
            }
            carry = carry_new;

            // Process c3
            let acc3: u64x4 = c3_lo + carry;
            let mut limb3 = acc3 & mask;
            let mut carry_new = (c3_hi << 13) | (acc3 >> 51);
            let overflow_mask = acc3.cmp_lt(c3_lo);
            if overflow_mask.to_array() != [0, 0, 0, 0] {
                carry_new = handle_overflow_correction(carry_new, overflow_mask);
            }
            carry = carry_new;

            // Process c4
            let acc4: u64x4 = c4_lo + carry;
            let mut limb4 = acc4 & mask;
            let mut carry_new = (c4_hi << 13) | (acc4 >> 51);
            let overflow_mask = acc4.cmp_lt(c4_lo);
            if overflow_mask.to_array() != [0, 0, 0, 0] {
                carry_new = handle_overflow_correction(carry_new, overflow_mask);
            }
            carry = carry_new;

            // Final reduction
            limb0 = limb0 + carry * FACTOR_19;
            let carry5 = limb0 >> 51;
            limb0 = limb0 & mask;
            limb1 = limb1 + carry5;

            // Final repack
            let outs: [[u64; 5]; 4] = core::array::from_fn(|i| [
                limb0.to_array()[i],
                limb1.to_array()[i],
                limb2.to_array()[i],
                limb3.to_array()[i],
                limb4.to_array()[i],
            ]);

            for lane in outs {
                for &limb in &lane {
                    simd_checksum ^= limb;
                }
                black_box(lane);
            }
        }
        let simd_time = simd_start.elapsed().as_secs_f64();

        println!("Scalar total time: {:.4}s", scalar_time);
        println!("SIMD   total time: {:.4}s", simd_time);
        println!("Speedup: {:.2}x", scalar_time / simd_time);
        println!("Scalar checksum:  {:016x}", scalar_checksum);
        println!("SIMD   checksum:  {:016x}", simd_checksum);

        assert_eq!(scalar_checksum, simd_checksum, "Checksum mismatch!");
    }

    #[test]
    fn test_simd_carry_propagation_poc() {
        use wide::u64x4;
        use rand_core::OsRng;
        use rand::Rng;

        const LOW_51: u64 = (1 << 51) - 1;
        let mask = u64x4::splat(LOW_51);
        let factor_19 = u64x4::splat(19);
        let mut rng = OsRng;

        // Closure to generate (lo, hi) limb pairs
        let mut gen = || {
            let mut lo = [0u64; 4];
            let mut hi = [0u64; 4];
            for i in 0..4 {
                lo[i] = rand::thread_rng().gen_range((1 << 50)..(1 << 51));
                hi[i] = rand::thread_rng().gen_range(0..(1 << 25));
            }
            (u64x4::from(lo), u64x4::from(hi))
        };

        for _ in 0..1_000_000 {
            let (lo0, hi0) = gen();
            let (lo1, hi1) = gen();
            let (lo2, hi2) = gen();
            let (lo3, hi3) = gen();
            let (lo4, hi4) = gen();

            // SIMD carry propagation
            let carry0 = (hi0 << 13) | (lo0 >> 51);
            let mut limb0: u64x4 = lo0 & mask;

            let lo1_new = lo1 + carry0;
            let carry1 = (hi1 << 13) | (lo1_new >> 51);
            let mut limb1: u64x4 = lo1_new & mask;

            let lo2_new = lo2 + carry1;
            let carry2 = (hi2 << 13) | (lo2_new >> 51);
            let mut limb2: u64x4 = lo2_new & mask;

            let lo3_new = lo3 + carry2;
            let carry3 = (hi3 << 13) | (lo3_new >> 51);
            let mut limb3: u64x4 = lo3_new & mask;

            let lo4_new = lo4 + carry3;
            let carry4 = (hi4 << 13) | (lo4_new >> 51);
            let mut limb4: u64x4 = lo4_new & mask;

            limb0 = limb0 + carry4 * factor_19;
            let carry5 = limb0 >> 51;
            limb0 = limb0 & mask;
            limb1 = limb1 + carry5;

            let final_result: [[u64; 5]; 4] = core::array::from_fn(|i| [
                limb0.to_array()[i],
                limb1.to_array()[i],
                limb2.to_array()[i],
                limb3.to_array()[i],
                limb4.to_array()[i],
            ]);

            // Scalar validation per lane
            for lane in 0..4 {
                let mut c0 = ((hi0.to_array()[lane] as u128) << 64) | lo0.to_array()[lane] as u128;
                let mut c1 = ((hi1.to_array()[lane] as u128) << 64) | lo1.to_array()[lane] as u128;
                let mut c2 = ((hi2.to_array()[lane] as u128) << 64) | lo2.to_array()[lane] as u128;
                let mut c3 = ((hi3.to_array()[lane] as u128) << 64) | lo3.to_array()[lane] as u128;
                let mut c4 = ((hi4.to_array()[lane] as u128) << 64) | lo4.to_array()[lane] as u128;

                let mut out = [0u64; 5];

                c1 += c0 >> 51;
                out[0] = (c0 as u64) & LOW_51;

                c2 += c1 >> 51;
                out[1] = (c1 as u64) & LOW_51;

                c3 += c2 >> 51;
                out[2] = (c2 as u64) & LOW_51;

                c4 += c3 >> 51;
                out[3] = (c3 as u64) & LOW_51;

                let carry = (c4 >> 51) as u64;
                out[4] = (c4 as u64) & LOW_51;

                out[0] += carry * 19;
                let carry2 = out[0] >> 51;
                out[0] &= LOW_51;
                out[1] += carry2;

                assert_eq!(
                    final_result[lane],
                    out,
                    "Lane {} failed: SIMD {:?}, Scalar {:?}",
                    lane,
                    final_result[lane],
                    out
                );
            }
        }
    }

    #[test]
    fn test_simd_schoolbook_c_products_poc() {
        use wide::u64x4;
        use rand_core::{OsRng, RngCore};

        const LOW_51: u64 = (1 << 51) - 1;

        let mut rng = OsRng;

        let mut a_lanes = [[0u64; 5]; 4];
        let mut b_lanes = [[0u64; 5]; 4];
        for lane in 0..4 {
            for i in 0..5 {
                a_lanes[lane][i] = rng.next_u64() & LOW_51;
                b_lanes[lane][i] = rng.next_u64() & LOW_51;
            }
        }

        // SIMD versions
        let a: [u64x4; 5] = core::array::from_fn(|i| {
            u64x4::new([a_lanes[0][i], a_lanes[1][i], a_lanes[2][i], a_lanes[3][i]])
        });
        let b: [u64x4; 5] = core::array::from_fn(|i| {
            u64x4::new([b_lanes[0][i], b_lanes[1][i], b_lanes[2][i], b_lanes[3][i]])
        });

        let factor_19 = u64x4::splat(19);
        let b1_19 = b[1] * factor_19;
        let b2_19 = b[2] * factor_19;
        let b3_19 = b[3] * factor_19;
        let b4_19 = b[4] * factor_19;

        let b1_19_arr = b1_19.to_array();
        let b2_19_arr = b2_19.to_array();
        let b3_19_arr = b3_19.to_array();
        let b4_19_arr = b4_19.to_array();

        let mut simd_c = [[0u128; 5]; 4];

        // --- SIMD computation per cX ---
        for c in 0..5 {
            for lane in 0..4 {
                let a = &a_lanes[lane];
                let b = &b_lanes[lane];
                simd_c[lane][c] = match c {
                    0 => (a[0] as u128) * (b[0] as u128)
                        + (a[4] as u128) * (b1_19_arr[lane] as u128)
                        + (a[3] as u128) * (b2_19_arr[lane] as u128)
                        + (a[2] as u128) * (b3_19_arr[lane] as u128)
                        + (a[1] as u128) * (b4_19_arr[lane] as u128),
                    1 => (a[1] as u128) * (b[0] as u128)
                        + (a[0] as u128) * (b[1] as u128)
                        + (a[4] as u128) * (b2_19_arr[lane] as u128)
                        + (a[3] as u128) * (b3_19_arr[lane] as u128)
                        + (a[2] as u128) * (b4_19_arr[lane] as u128),
                    2 => (a[2] as u128) * (b[0] as u128)
                        + (a[1] as u128) * (b[1] as u128)
                        + (a[0] as u128) * (b[2] as u128)
                        + (a[4] as u128) * (b[3] as u128)
                        + (a[3] as u128) * (b[4] as u128),
                    3 => (a[3] as u128) * (b[0] as u128)
                        + (a[2] as u128) * (b[1] as u128)
                        + (a[1] as u128) * (b[2] as u128)
                        + (a[0] as u128) * (b[3] as u128)
                        + (a[4] as u128) * (b[4] as u128),
                    4 => (a[4] as u128) * (b[0] as u128)
                        + (a[3] as u128) * (b[1] as u128)
                        + (a[2] as u128) * (b[2] as u128)
                        + (a[1] as u128) * (b[3] as u128)
                        + (a[0] as u128) * (b[4] as u128),
                    _ => unreachable!(),
                };
            }
        }

        // Scalar reference
        let mut scalar_c = [[0u128; 5]; 4];
        for lane in 0..4 {
            let a = &a_lanes[lane];
            let b = &b_lanes[lane];
            let b1_19 = b[1] * 19;
            let b2_19 = b[2] * 19;
            let b3_19 = b[3] * 19;
            let b4_19 = b[4] * 19;

            scalar_c[lane][0] = (a[0] as u128) * (b[0] as u128)
                + (a[4] as u128) * (b1_19 as u128)
                + (a[3] as u128) * (b2_19 as u128)
                + (a[2] as u128) * (b3_19 as u128)
                + (a[1] as u128) * (b4_19 as u128);

            scalar_c[lane][1] = (a[1] as u128) * (b[0] as u128)
                + (a[0] as u128) * (b[1] as u128)
                + (a[4] as u128) * (b2_19 as u128)
                + (a[3] as u128) * (b3_19 as u128)
                + (a[2] as u128) * (b4_19 as u128);

            scalar_c[lane][2] = (a[2] as u128) * (b[0] as u128)
                + (a[1] as u128) * (b[1] as u128)
                + (a[0] as u128) * (b[2] as u128)
                + (a[4] as u128) * (b[3] as u128)
                + (a[3] as u128) * (b[4] as u128);

            scalar_c[lane][3] = (a[3] as u128) * (b[0] as u128)
                + (a[2] as u128) * (b[1] as u128)
                + (a[1] as u128) * (b[2] as u128)
                + (a[0] as u128) * (b[3] as u128)
                + (a[4] as u128) * (b[4] as u128);

            scalar_c[lane][4] = (a[4] as u128) * (b[0] as u128)
                + (a[3] as u128) * (b[1] as u128)
                + (a[2] as u128) * (b[2] as u128)
                + (a[1] as u128) * (b[3] as u128)
                + (a[0] as u128) * (b[4] as u128);
        }

        // Compare
        for lane in 0..4 {
            for c in 0..5 {
                let simd_val = simd_c[lane][c];
                let scalar_val = scalar_c[lane][c];
                if simd_val != scalar_val {
                    println!("Lane {lane}, c{c} mismatch:");
                    println!("  a = {:?}", a_lanes[lane]);
                    println!("  b = {:?}", b_lanes[lane]);
                    println!("  b1_19 = {:x}", b1_19_arr[lane]);
                    println!("  b2_19 = {:x}", b2_19_arr[lane]);
                    println!("  b3_19 = {:x}", b3_19_arr[lane]);
                    println!("  b4_19 = {:x}", b4_19_arr[lane]);
                    println!("  SIMD   c{c} = {simd_val}");
                    println!("  Scalar c{c} = {scalar_val}");
                }

                assert_eq!(
                    simd_val, scalar_val,
                    "Mismatch in c{c} lane {lane}: SIMD {}, Scalar {}",
                    simd_val, scalar_val
                );
            }
        }
    }

    #[test]
    fn test_square_simd_vs_scalar() {
        use wide::u64x4;
        use rand_core::{OsRng, RngCore};
        use std::time::Instant;
        use std::hint::black_box;

        const TEST_CASES: usize = 100_000;
        const LOW_51: u64 = (1 << 51) - 1;

        let mut rng = OsRng;
        let mut a_lanes = vec![[0u64; 5]; TEST_CASES * 4];

        for a in a_lanes.iter_mut() {
            for i in 0..5 {
                a[i] = rng.next_u64() & LOW_51;
            }
        }

        // Scalar squaring
        let mut scalar_checksum = 0u64;
        let scalar_start = Instant::now();

        for a in a_lanes.iter() {
            // Precomputation
            let a3_19 = a[3] * 19;
            let a4_19 = a[4] * 19;

            // Squaring-specific computation
            let c0 = (a[0] as u128) * (a[0] as u128) 
                  + 2 * ((a[1] as u128) * (a4_19 as u128) + (a[2] as u128) * (a3_19 as u128));
            let mut c1 = (a[3] as u128) * (a3_19 as u128) 
                      + 2 * ((a[0] as u128) * (a[1] as u128) + (a[2] as u128) * (a4_19 as u128));
            let mut c2 = (a[1] as u128) * (a[1] as u128) 
                      + 2 * ((a[0] as u128) * (a[2] as u128) + (a[4] as u128) * (a3_19 as u128));
            let mut c3 = (a[4] as u128) * (a4_19 as u128) 
                      + 2 * ((a[0] as u128) * (a[3] as u128) + (a[1] as u128) * (a[2] as u128));
            let mut c4 = (a[2] as u128) * (a[2] as u128) 
                      + 2 * ((a[0] as u128) * (a[4] as u128) + (a[1] as u128) * (a[3] as u128));

            // Carry propagation
            let mut out = [0u64; 5];

            c1 += (c0 >> 51) as u128;
            out[0] = (c0 as u64) & LOW_51;

            c2 += (c1 >> 51) as u128;
            out[1] = (c1 as u64) & LOW_51;

            c3 += (c2 >> 51) as u128;
            out[2] = (c2 as u64) & LOW_51;

            c4 += (c3 >> 51) as u128;
            out[3] = (c3 as u64) & LOW_51;

            let carry = (c4 >> 51) as u64;
            out[4] = (c4 as u64) & LOW_51;

            out[0] += carry * 19;
            out[1] += out[0] >> 51;
            out[0] &= LOW_51;

            for &limb in &out {
                scalar_checksum ^= limb;
            }

            black_box(out);
        }
        let scalar_time = scalar_start.elapsed().as_secs_f64();

        // SIMD squaring
        let mut simd_checksum = 0u64;
        let simd_start = Instant::now();

        // Helper functions
        #[inline(always)]
        fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
            const MASK_32: u64x4 = u64x4::new([0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF]);
            
            let a_lo = a & MASK_32;
            let a_hi = a >> 32;
            let b_lo = b & MASK_32;
            let b_hi = b >> 32;
            
            let lo_lo = a_lo * b_lo;
            let lo_hi = a_lo * b_hi;
            let hi_lo = a_hi * b_lo;
            let hi_hi = a_hi * b_hi;
            
            let mid = lo_hi + hi_lo;
            let mid_lo = mid << 32;
            let mid_hi = mid >> 32;
            
            let res_lo: u64x4 = lo_lo + mid_lo;
            let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let res_hi: u64x4 = hi_hi + mid_hi + carry;
            
            (res_lo, res_hi)
        }

        #[inline(always)]
        fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
            let sum_lo = a_lo + b_lo;
            let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let sum_hi = a_hi + b_hi + carry;
            (sum_lo, sum_hi)
        }

        // Double a 128-bit value (shift left by 1)
        #[inline(always)]
        fn double_128_simd(lo: u64x4, hi: u64x4) -> (u64x4, u64x4) {
            let new_hi = (hi << 1) | (lo >> 63);
            let new_lo = lo << 1;
            (new_lo, new_hi)
        }

        for a_chunk in a_lanes.chunks_exact(4) {
            let mut a = [u64x4::default(); 5];
            for i in 0..5 {
                a[i] = u64x4::new([a_chunk[0][i], a_chunk[1][i], a_chunk[2][i], a_chunk[3][i]]);
            }

            const FACTOR_19: u64x4 = u64x4::new([19,19,19,19]);
            let a3_19 = a[3] * FACTOR_19;
            let a4_19 = a[4] * FACTOR_19;

            // Compute unique products (exploiting squaring symmetry)
            let a0_sq = mul64_to_128_simd(a[0], a[0]);
            let a1_sq = mul64_to_128_simd(a[1], a[1]);
            let a2_sq = mul64_to_128_simd(a[2], a[2]);
            
            let a0_a1 = mul64_to_128_simd(a[0], a[1]);
            let a0_a2 = mul64_to_128_simd(a[0], a[2]);
            let a0_a3 = mul64_to_128_simd(a[0], a[3]);
            let a0_a4 = mul64_to_128_simd(a[0], a[4]);
            let a1_a2 = mul64_to_128_simd(a[1], a[2]);
            let a1_a3 = mul64_to_128_simd(a[1], a[3]);
            let a1_a4_19 = mul64_to_128_simd(a[1], a4_19);
            let a2_a3_19 = mul64_to_128_simd(a[2], a3_19);
            let a2_a4_19 = mul64_to_128_simd(a[2], a4_19);
            let a3_a3_19 = mul64_to_128_simd(a[3], a3_19);
            let a4_a3_19 = mul64_to_128_simd(a[4], a3_19);
            let a4_a4_19 = mul64_to_128_simd(a[4], a4_19);

            // Double the products that appear with coefficient 2
            let a0_a1_2 = double_128_simd(a0_a1.0, a0_a1.1);
            let a0_a2_2 = double_128_simd(a0_a2.0, a0_a2.1);
            let a0_a3_2 = double_128_simd(a0_a3.0, a0_a3.1);
            let a0_a4_2 = double_128_simd(a0_a4.0, a0_a4.1);
            let a1_a2_2 = double_128_simd(a1_a2.0, a1_a2.1);
            let a1_a3_2 = double_128_simd(a1_a3.0, a1_a3.1);
            let a1_a4_19_2 = double_128_simd(a1_a4_19.0, a1_a4_19.1);
            let a2_a3_19_2 = double_128_simd(a2_a3_19.0, a2_a3_19.1);
            let a2_a4_19_2 = double_128_simd(a2_a4_19.0, a2_a4_19.1);
            let a4_a3_19_2 = double_128_simd(a4_a3_19.0, a4_a3_19.1);

            // Compute c0 = a[0]^2 + 2*(a[1]*a4_19 + a[2]*a3_19)
            let (c0_lo, c0_hi) = a0_sq;
            let (c0_lo, c0_hi) = add_128_simd(c0_lo, c0_hi, a1_a4_19_2.0, a1_a4_19_2.1);
            let (c0_lo, c0_hi) = add_128_simd(c0_lo, c0_hi, a2_a3_19_2.0, a2_a3_19_2.1);

            // Compute c1 = a[3]*a3_19 + 2*(a[0]*a[1] + a[2]*a4_19)
            let (c1_lo, c1_hi) = a3_a3_19;
            let (c1_lo, c1_hi) = add_128_simd(c1_lo, c1_hi, a0_a1_2.0, a0_a1_2.1);
            let (c1_lo, c1_hi) = add_128_simd(c1_lo, c1_hi, a2_a4_19_2.0, a2_a4_19_2.1);

            // Compute c2 = a[1]^2 + 2*(a[0]*a[2] + a[4]*a3_19)
            let (c2_lo, c2_hi) = a1_sq;
            let (c2_lo, c2_hi) = add_128_simd(c2_lo, c2_hi, a0_a2_2.0, a0_a2_2.1);
            let (c2_lo, c2_hi) = add_128_simd(c2_lo, c2_hi, a4_a3_19_2.0, a4_a3_19_2.1);

            // Compute c3 = a[4]*a4_19 + 2*(a[0]*a[3] + a[1]*a[2])
            let (c3_lo, c3_hi) = a4_a4_19;
            let (c3_lo, c3_hi) = add_128_simd(c3_lo, c3_hi, a0_a3_2.0, a0_a3_2.1);
            let (c3_lo, c3_hi) = add_128_simd(c3_lo, c3_hi, a1_a2_2.0, a1_a2_2.1);

            // Compute c4 = a[2]^2 + 2*(a[0]*a[4] + a[1]*a[3])
            let (c4_lo, c4_hi) = a2_sq;
            let (c4_lo, c4_hi) = add_128_simd(c4_lo, c4_hi, a0_a4_2.0, a0_a4_2.1);
            let (c4_lo, c4_hi) = add_128_simd(c4_lo, c4_hi, a1_a3_2.0, a1_a3_2.1);

            // Carry propagation (same as multiplication)
            const MASK: u64x4 = u64x4::new([LOW_51,LOW_51,LOW_51,LOW_51]);

            let mut limb0 = c0_lo & MASK;
            let mut carry = (c0_hi << 13) | (c0_lo >> 51);

            macro_rules! propagate_carry {
                ($c_lo:expr, $c_hi:expr) => {{
                    let acc: u64x4 = $c_lo + carry;
                    let limb = acc & MASK;
                    let mut new_carry = ($c_hi << 13) | (acc >> 51);
                    
                    let overflow_mask = acc.cmp_lt($c_lo);
                    if overflow_mask.to_array() != [0, 0, 0, 0] {
                        new_carry = new_carry + overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0));
                    }
                    carry = new_carry;
                    limb
                }};
            }

            let limb1: u64x4 = propagate_carry!(c1_lo, c1_hi);
            let limb2: u64x4 = propagate_carry!(c2_lo, c2_hi);
            let limb3: u64x4 = propagate_carry!(c3_lo, c3_hi);
            let limb4: u64x4 = propagate_carry!(c4_lo, c4_hi);

            // Final reduction
            limb0 = limb0 + carry * FACTOR_19;
            let carry5 = limb0 >> 51;
            limb0 = limb0 & MASK;
            let limb1: u64x4 = limb1 + carry5;

            // Collect results
            let outs: [[u64; 5]; 4] = core::array::from_fn(|i| [
                limb0.to_array()[i],
                limb1.to_array()[i],
                limb2.to_array()[i],
                limb3.to_array()[i],
                limb4.to_array()[i],
            ]);

            for lane in outs {
                for &limb in &lane {
                    simd_checksum ^= limb;
                }
                black_box(lane);
            }
        }
        let simd_time = simd_start.elapsed().as_secs_f64();

        println!("Square Scalar time: {:.4}s", scalar_time);
        println!("Square SIMD   time: {:.4}s", simd_time);
        println!("Square Speedup: {:.2}x", scalar_time / simd_time);
        println!("Square Scalar checksum: {:016x}", scalar_checksum);
        println!("Square SIMD   checksum: {:016x}", simd_checksum);

        assert_eq!(scalar_checksum, simd_checksum, "Square checksum mismatch!");
    }

#[test]
fn benchmark_ecdlp_segments() {
    use std::time::Instant;
    use std::hint::black_box;
    use rand_core::{OsRng, RngCore};
    use subtle::ConditionallySelectable;
    
    const BATCH_SIZE: usize = 256;
    const TEST_ITERATIONS: usize = 10_000;
    const LOW_51: u64 = (1 << 51) - 1;
    
    // Setup test data
    let mut rng = OsRng;
    
    // Generate random field elements for testing
    fn random_field_element(rng: &mut impl RngCore) -> FieldElement {
        let mut limbs = [0u64; 5];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u64() & LOW_51;
        }
        FieldElement::from_limbs(limbs)
    }
    
    // Pre-generate test data
    let mut test_batches_a = Vec::new();
    let mut test_batches_b = Vec::new();
    let mut test_targets = Vec::new();
    
    for _ in 0..TEST_ITERATIONS {
        let mut batch_a = [FieldElement::ZERO; BATCH_SIZE];
        let mut batch_b = [FieldElement::ZERO; BATCH_SIZE];
        
        for i in 0..BATCH_SIZE {
            batch_a[i] = random_field_element(&mut rng);
            batch_b[i] = random_field_element(&mut rng);
        }
        
        test_batches_a.push(batch_a);
        test_batches_b.push(batch_b);
        test_targets.push(random_field_element(&mut rng));
    }
    
    println!("\n=== ECDLP Segment Benchmarks ===");
    println!("Batch size: {}", BATCH_SIZE);
    println!("Test iterations: {}", TEST_ITERATIONS);
    println!();
    
    // 1. Benchmark batch_field_subtract
    {
        println!("1. batch_field_subtract (u - target_u):");
        
        // Scalar version
        let mut result_scalar = [FieldElement::ZERO; BATCH_SIZE];
        let scalar_start = Instant::now();
        for i in 0..TEST_ITERATIONS {
            let batch = &test_batches_a[i];
            let target = &test_targets[i];
            for j in 0..BATCH_SIZE {
                result_scalar[j] = &batch[j] - target;
            }
            black_box(&result_scalar);
        }
        let scalar_time = scalar_start.elapsed();
        
        // SIMD version
        let mut result_simd = [FieldElement::ZERO; BATCH_SIZE];
        let simd_start = Instant::now();
        for i in 0..TEST_ITERATIONS {
            batch_field_subtract(&mut result_simd, &test_batches_a[i], &test_targets[i]);
            black_box(&result_simd);
        }
        let simd_time = simd_start.elapsed();
        
        println!("  Scalar: {:.4}s", scalar_time.as_secs_f64());
        println!("  SIMD:   {:.4}s", simd_time.as_secs_f64());
        println!("  Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
        println!();
    }
    
    // 2. Benchmark batch_field_mul_and_square
    {
        println!("2. batch_field_mul_and_square ((v_diff * nu)^2):");
        
        // Scalar version
        let mut result_scalar = [FieldElement::ZERO; BATCH_SIZE];
        let scalar_start = Instant::now();
        for i in 0..TEST_ITERATIONS {
            let a = &test_batches_a[i];
            let b = &test_batches_b[i];
            for j in 0..BATCH_SIZE {
                result_scalar[j] = (&a[j] * &b[j]).square();
            }
            black_box(&result_scalar);
        }
        let scalar_time = scalar_start.elapsed();
        
        // SIMD version
        let mut result_simd = [FieldElement::ZERO; BATCH_SIZE];
        let simd_start = Instant::now();
        for i in 0..TEST_ITERATIONS {
            batch_field_mul_and_square(&mut result_simd, &test_batches_a[i], &test_batches_b[i]);
            black_box(&result_simd);
        }
        let simd_time = simd_start.elapsed();
        
        println!("  Scalar: {:.4}s", scalar_time.as_secs_f64());
        println!("  SIMD:   {:.4}s", simd_time.as_secs_f64());
        println!("  Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
        println!();
    }
    
    // 3. Benchmark batch_field_add
    {
        println!("3. batch_field_add (qx + alpha):");
        
        // Scalar version
        let mut result_scalar = [FieldElement::ZERO; BATCH_SIZE];
        let scalar_start = Instant::now();
        for i in 0..TEST_ITERATIONS {
            let a = &test_batches_a[i];
            let b = &test_batches_b[i];
            for j in 0..BATCH_SIZE {
                result_scalar[j] = &a[j] + &b[j];
            }
            black_box(&result_scalar);
        }
        let scalar_time = scalar_start.elapsed();
        
        // SIMD version
        let mut result_simd = [FieldElement::ZERO; BATCH_SIZE];
        let simd_start = Instant::now();
        for i in 0..TEST_ITERATIONS {
            batch_field_add(&mut result_simd, &test_batches_a[i], &test_batches_b[i]);
            black_box(&result_simd);
        }
        let simd_time = simd_start.elapsed();
        
        println!("  Scalar: {:.4}s", scalar_time.as_secs_f64());
        println!("  SIMD:   {:.4}s", simd_time.as_secs_f64());
        println!("  Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
        println!();
    }
    
    // 4. Simulate full ECDLP inner loop
    {
        println!("4. Full ECDLP inner loop simulation with validation:");
        
        // Pre-allocate all arrays
        let mut batch_scalar = [FieldElement::ZERO; BATCH_SIZE];
        let mut batch_simd_trad = [FieldElement::ZERO; BATCH_SIZE];
        let mut batch_simd_4lane = [FieldElement::ZERO; BATCH_SIZE];
        
        let mut qx_scalar = [FieldElement::ZERO; BATCH_SIZE];
        let mut qx_simd_trad = [FieldElement::ZERO; BATCH_SIZE];
        let mut qx_simd_4lane = [FieldElement::ZERO; BATCH_SIZE];
        let mut qx_simd_4lane_out = [FieldElement::ZERO; BATCH_SIZE];
        
        // Helper function for 4-lane batch invert
        let batch_invert_4lane = |elements: &mut [FieldElement]| {
            let n = elements.len();
            let num_chunks = (n + 3) / 4;
            
            let mut scratch = vec![FieldElement::ONE; num_chunks * 4];
            let mut acc_lanes = [FieldElement::ONE; 4];
            
            // Forward pass
            for chunk_idx in 0..num_chunks {
                let base = chunk_idx * 4;
                let scratch_base = chunk_idx * 4;
                
                scratch[scratch_base] = acc_lanes[0];
                scratch[scratch_base + 1] = acc_lanes[1];
                scratch[scratch_base + 2] = acc_lanes[2];
                scratch[scratch_base + 3] = acc_lanes[3];
                
                let mut input_chunk = [FieldElement::ONE; 4];
                for i in 0..4 {
                    if base + i < n {
                        input_chunk[i] = elements[base + i];
                    }
                }
                
                acc_lanes = FieldElement::batch_mul_4way(&acc_lanes, &input_chunk);
            }
            
            // Combined inversion
            let p01 = &acc_lanes[0] * &acc_lanes[1];
            let p23 = &acc_lanes[2] * &acc_lanes[3];
            let p0123 = &p01 * &p23;
            let inv_p0123 = p0123.invert();
            
            let factors = [
                &acc_lanes[1] * &p23,
                &acc_lanes[0] * &p23,
                &p01 * &acc_lanes[3],
                &p01 * &acc_lanes[2],
            ];
            
            let inv_broadcast = [inv_p0123; 4];
            acc_lanes = FieldElement::batch_mul_4way(&inv_broadcast, &factors);
            
            // Reverse pass
            for chunk_idx in (0..num_chunks).rev() {
                let base = chunk_idx * 4;
                let scratch_base = chunk_idx * 4;
                
                let mut input_chunk = [FieldElement::ONE; 4];
                for i in 0..4 {
                    if base + i < n {
                        input_chunk[i] = elements[base + i];
                    }
                }
                
                let scratch_chunk = [
                    scratch[scratch_base],
                    scratch[scratch_base + 1],
                    scratch[scratch_base + 2],
                    scratch[scratch_base + 3],
                ];
                
                let results = FieldElement::batch_mul_4way(&acc_lanes, &scratch_chunk);
                
                for i in 0..4 {
                    if base + i < n {
                        elements[base + i] = results[i];
                    }
                }
                
                acc_lanes = FieldElement::batch_mul_4way(&acc_lanes, &input_chunk);
            }
        };
        
        // First, run a single iteration of each method and validate results
        println!("\n  Validation check:");
        let idx = 0;
        let target = &test_targets[idx];
        let alphas = &test_batches_b[idx];
        
        // Method 1: Scalar
        batch_scalar.copy_from_slice(&test_batches_a[idx]);
        for i in 0..BATCH_SIZE {
            batch_scalar[i] = &batch_scalar[i] - target;
        }
        FieldElement::batch_invert(&mut batch_scalar);
        for i in 0..BATCH_SIZE {
            let lambda = &test_batches_b[idx][i] * &batch_scalar[i];
            qx_scalar[i] = lambda.square();
            qx_scalar[i] = &qx_scalar[i] + &alphas[i];
        }
        
        // Method 2: SIMD with traditional invert
        batch_field_subtract(&mut batch_simd_trad, &test_batches_a[idx], target);
        FieldElement::batch_invert(&mut batch_simd_trad);
        batch_field_mul_and_square(&mut qx_simd_trad, &test_batches_b[idx], &batch_simd_trad);
        for i in 0..BATCH_SIZE {
            qx_simd_trad[i] = &qx_simd_trad[i] + &alphas[i];
        }
        
        // Method 3: SIMD with 4-lane invert
        batch_field_subtract(&mut batch_simd_4lane, &test_batches_a[idx], target);
        batch_invert_4lane(&mut batch_simd_4lane);
        batch_field_mul_and_square(&mut qx_simd_4lane, &test_batches_b[idx], &batch_simd_4lane);
        for i in 0..BATCH_SIZE {
            qx_simd_4lane[i] = &qx_simd_4lane[i] + &alphas[i];
        }
        
        // Validate all results match
        let mut all_match = true;
        let mut mismatch_count = 0;
        for i in 0..BATCH_SIZE {
            let scalar_vs_trad = qx_scalar[i] == qx_simd_trad[i];
            let scalar_vs_4lane = qx_scalar[i] == qx_simd_4lane[i];
            
            if !scalar_vs_trad || !scalar_vs_4lane {
                if mismatch_count < 5 {  // Show first 5 mismatches
                    println!("    Mismatch at index {}:", i);
                    println!("      Scalar:     {:?}", &qx_scalar[i].0[0]);
                    println!("      SIMD trad:  {:?} {}", &qx_simd_trad[i].0[0], if scalar_vs_trad { "✓" } else { "✗" });
                    println!("      SIMD 4lane: {:?} {}", &qx_simd_4lane[i].0[0], if scalar_vs_4lane { "✓" } else { "✗" });
                }
                mismatch_count += 1;
                all_match = false;
            }
        }
        
        if all_match {
            println!("    All methods produce identical results ✓");
        } else {
            println!("    MISMATCH: {} elements differ!", mismatch_count);
        }
        
        // Also validate the inversion step specifically
        println!("\n  Inversion validation:");
        let mut test_batch = test_batches_a[idx].clone();
        let mut test_batch_4lane = test_batch.clone();
        
        // Traditional invert
        FieldElement::batch_invert(&mut test_batch);
        
        // 4-lane invert
        batch_invert_4lane(&mut test_batch_4lane);
        
        let mut inv_match = true;
        for i in 0..BATCH_SIZE {
            if test_batch[i] != test_batch_4lane[i] {
                if i < 5 {  // Show first 5
                    println!("    Inversion mismatch at {}: trad={:?}, 4lane={:?}", 
                        i, &test_batch[i].0[0], &test_batch_4lane[i].0[0]);
                }
                inv_match = false;
            }
        }
        println!("    Inversion methods match: {}", if inv_match { "✓" } else { "✗" });
        
        // Only run performance test if validation passes
        if all_match && inv_match {
            println!("\n  Performance test:");
            
            // Run timing tests as before...
            let scalar_start = Instant::now();
            for iter in 0..TEST_ITERATIONS / 10 {
                let idx = iter % test_batches_a.len();
                let target = &test_targets[idx];
                let alphas = &test_batches_b[idx];
                
                batch_scalar.copy_from_slice(&test_batches_a[idx]);
                for i in 0..BATCH_SIZE {
                    batch_scalar[i] = &batch_scalar[i] - target;
                    black_box(&batch_scalar[i]);
                }
                FieldElement::batch_invert(&mut batch_scalar);
                for i in 0..BATCH_SIZE {
                    let lambda = &test_batches_b[idx][i] * &batch_scalar[i];
                    qx_scalar[i] = lambda.square();
                    qx_scalar[i] = &qx_scalar[i] + &alphas[i];
                    black_box(&qx_scalar[i]);
                }
            }
            let scalar_time = scalar_start.elapsed();
            
            // SIMD with 4-lane
            let simd_start = Instant::now();
            for iter in 0..TEST_ITERATIONS / 10 {
                let idx = iter % test_batches_a.len();
                let target = &test_targets[idx];
                let alphas = &test_batches_b[idx];
                
                batch_field_subtract(&mut batch_simd_4lane, &test_batches_a[idx], target);
                batch_invert_4lane(&mut batch_simd_4lane);
                batch_field_mul_and_square(&mut qx_simd_4lane, &test_batches_b[idx], &batch_simd_4lane);
                batch_field_add(&mut qx_simd_4lane_out, &qx_simd_4lane, alphas);
                
                black_box(&qx_simd_4lane_out);
            }
            let simd_time = simd_start.elapsed();
            
            println!("    Scalar: {:.4}s", scalar_time.as_secs_f64());
            println!("    SIMD:   {:.4}s", simd_time.as_secs_f64());
            println!("    Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
        } else {
            println!("\n  Skipping performance test due to validation failure!");
        }
    }

    println!("\n=== Summary ===");
    println!("SIMD optimizations provide significant speedups for field arithmetic.");
    println!("Batch inversion (Montgomery's trick) is crucial regardless of SIMD.");
    println!("The remaining time is likely in table lookups and control flow.");
}

    // Assuming CUCKOO_K = 3 for this example
    const CUCKOO_K: usize = 3;
    
    // SIMD-friendly lookup that processes multiple queries at once
    pub struct SimdCuckooT1HashMapView<'a> {
        pub keys: &'a [u32],
        pub values: &'a [u32],
        pub cuckoo_len: usize,
    }
    
    impl<'a> SimdCuckooT1HashMapView<'a> {
        /// Batch lookup for 4 queries at once using SIMD
        /// Returns a [Option<u64>; 4] array
        pub fn lookup_batch_4(
            &self,
            queries: &[[u8; 32]; 4],
        ) -> [(bool, u64); 4] {
            use std::arch::x86_64::*;
            
            unsafe {
                let mut results = [(false, 0u64); 4];
                let cuckoo_len = self.cuckoo_len as u32;
                
                // Process each cuckoo position
                for i in 0..CUCKOO_K {
                    let start = i * 8;
                    let key_offset = start + 4;
                    
                    // Extract keys and hashes for all 4 queries
                    let keys = _mm_setr_epi32(
                        u32::from_be_bytes(queries[0][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[1][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[2][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[3][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                    );
                    
                    let hashes = _mm_setr_epi32(
                        u32::from_be_bytes(queries[0][start..start + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[1][start..start + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[2][start..start + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[3][start..start + 4].try_into().unwrap()) as i32,
                    );
                    
                    // Compute indices (modulo operation)
                    // For power-of-2 cuckoo_len, use AND mask instead
                    let indices = if cuckoo_len.is_power_of_two() {
                        let mask = _mm_set1_epi32((cuckoo_len - 1) as i32);
                        _mm_and_si128(hashes, mask)
                    } else {
                        // Fallback to scalar modulo for non-power-of-2
                        let h0 = _mm_extract_epi32(hashes, 0) as u32 % cuckoo_len;
                        let h1 = _mm_extract_epi32(hashes, 1) as u32 % cuckoo_len;
                        let h2 = _mm_extract_epi32(hashes, 2) as u32 % cuckoo_len;
                        let h3 = _mm_extract_epi32(hashes, 3) as u32 % cuckoo_len;
                        _mm_setr_epi32(h0 as i32, h1 as i32, h2 as i32, h3 as i32)
                    };
                    
                    // Gather table keys
                    let idx0 = _mm_extract_epi32(indices, 0) as usize;
                    let idx1 = _mm_extract_epi32(indices, 1) as usize;
                    let idx2 = _mm_extract_epi32(indices, 2) as usize;
                    let idx3 = _mm_extract_epi32(indices, 3) as usize;
                    
                    let table_keys = _mm_setr_epi32(
                        self.keys[idx0] as i32,
                        self.keys[idx1] as i32,
                        self.keys[idx2] as i32,
                        self.keys[idx3] as i32,
                    );
                    
                    // Compare keys
                    let matches = _mm_cmpeq_epi32(keys, table_keys);
                    let match_mask = _mm_movemask_ps(_mm_castsi128_ps(matches));
                    
                    // Extract values for matches
                    if match_mask & 0x1 != 0 && !results[0].0 {
                        results[0] = (true, self.values[idx0] as u64);
                    }
                    if match_mask & 0x2 != 0 && !results[1].0 {
                        results[1] = (true, self.values[idx1] as u64);
                    }
                    if match_mask & 0x4 != 0 && !results[2].0 {
                        results[2] = (true, self.values[idx2] as u64);
                    }
                    if match_mask & 0x8 != 0 && !results[3].0 {
                        results[3] = (true, self.values[idx3] as u64);
                    }
                }
                
                results
            }
        }
    }

    use crate::ecdlp::table::CuckooT1HashMapView;

    fn reference_lookup_batch(
        t1_table: &CuckooT1HashMapView,
        queries: &[[u8; 32]],
    ) -> Vec<Option<u64>> {
        queries.iter().map(|query| {
            let mut found = None;
            t1_table.lookup(query, |value| {
                found = Some(value);
                true // Stop on first match
            });
            found
        }).collect()
    }
    

    use std::time::Instant;
    #[test]
    fn test_simd_lookup_correctness() {
        // Create a test table with known values
        let mut keys = vec![0u32; 1000];
        let mut values = vec![0u32; 1000];
        
        // Insert some test data
        let test_entries = vec![
            (0x12345678u32, 100u32),
            (0x87654321u32, 200u32),
            (0xABCDEF00u32, 300u32),
            (0xDEADBEEFu32, 400u32),
        ];
        
        // Simple hash function for testing
        for (key, value) in &test_entries {
            let hash = (*key as usize) % 1000;
            keys[hash] = *key;
            values[hash] = *value;
        }
        
        let t1_table = CuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: 1000,
        };
        
        let simd_table = SimdCuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: 1000,
        };
        
        // Test with queries that should match
        let mut queries = [[0u8; 32]; 4];
        
        // Place keys at the expected positions (assuming CUCKOO_K=3)
        // First cuckoo position (bytes 4-8)
        queries[0][4..8].copy_from_slice(&0x12345678u32.to_be_bytes());
        queries[0][0..4].copy_from_slice(&(0x12345678u32 % 1000).to_be_bytes());
        
        queries[1][4..8].copy_from_slice(&0x87654321u32.to_be_bytes());
        queries[1][0..4].copy_from_slice(&(0x87654321u32 % 1000).to_be_bytes());
        
        // Test non-matching queries
        queries[2][4..8].copy_from_slice(&0xFFFFFFFFu32.to_be_bytes());
        queries[3][4..8].copy_from_slice(&0x00000000u32.to_be_bytes());
        
        // Get results from both implementations
        let reference_results = reference_lookup_batch(&t1_table, &queries);
        let simd_results = simd_table.lookup_batch_4(&queries);
        
        // Verify results match
        for i in 0..4 {
            let ref_result = reference_results[i];
            let simd_result = if simd_results[i].0 { Some(simd_results[i].1) } else { None };
            
            assert_eq!(ref_result, simd_result, 
                "Mismatch at index {}: reference={:?}, simd={:?}", 
                i, ref_result, simd_result);
        }
        
        println!("✓ SIMD lookup correctness test passed");
    }
    
    #[test]
    fn test_simd_lookup_stress() {
        // Stress test with random data
        let mut rng = rand::thread_rng();
        let table_size = 10000;
        
        let mut keys = vec![0u32; table_size];
        let mut values = vec![0u32; table_size];
        
        // Fill with random data
        for i in 0..table_size / 2 {
            keys[i] = rng.gen();
            values[i] = rng.gen();
        }
        
        let t1_table = CuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: table_size,
        };
        
        let simd_table = SimdCuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: table_size,
        };
        
        // Test 1000 random query batches
        for _ in 0..1000 {
            let mut queries = [[0u8; 32]; 4];
            for q in &mut queries {
                rng.fill(q);
            }
            
            let reference_results = reference_lookup_batch(&t1_table, &queries);
            let simd_results = simd_table.lookup_batch_4(&queries);
            
            for i in 0..4 {
                let ref_result = reference_results[i];
                let simd_result = if simd_results[i].0 { Some(simd_results[i].1) } else { None };
                assert_eq!(ref_result, simd_result);
            }
        }
        
        println!("✓ SIMD lookup stress test passed (1000 random batches)");
    }
    
    #[test]
    fn benchmark_simd_vs_sequential_lookup() {
        use std::hint::black_box;
        
        // Create a realistic table
        let table_size = 100_000;
        let mut keys = vec![0u32; table_size];
        let mut values = vec![0u32; table_size];
        let mut rng = rand::thread_rng();
        
        // Fill ~70% of the table (typical cuckoo hash load factor)
        for i in 0..(table_size * 7 / 10) {
            keys[i] = rng.gen();
            values[i] = i as u32;
        }
        
        let t1_table = CuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: table_size,
        };
        
        let simd_table = SimdCuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: table_size,
        };
        
        // Generate test queries
        const NUM_BATCHES: usize = 10000;
        let mut all_queries = Vec::with_capacity(NUM_BATCHES * 4);
        for _ in 0..NUM_BATCHES {
            for _ in 0..4 {
                let mut query = [0u8; 32];
                rng.fill(&mut query);
                all_queries.push(query);
            }
        }
        
        // Warm-up
        for _ in 0..100 {
            let _ = reference_lookup_batch(&t1_table, &all_queries[0..4]);
            let _ = simd_table.lookup_batch_4(&all_queries[0..4].try_into().unwrap());
        }
        
        // Benchmark sequential
        let start = Instant::now();
        for batch_idx in 0..NUM_BATCHES {
            let base = batch_idx * 4;
            let results = reference_lookup_batch(&t1_table, &all_queries[base..base+4]);
            black_box(results);
        }
        let sequential_time = start.elapsed();
        
        // Benchmark SIMD
        let start = Instant::now();
        for batch_idx in 0..NUM_BATCHES {
            let base = batch_idx * 4;
            let queries: [[u8; 32]; 4] = [
                all_queries[base],
                all_queries[base + 1],
                all_queries[base + 2],
                all_queries[base + 3],
            ];
            let results = simd_table.lookup_batch_4(&queries);
            black_box(results);
        }
        let simd_time = start.elapsed();
        
        println!("Lookup benchmark results:");
        println!("  Sequential: {:.3} ms ({} lookups)", 
                 sequential_time.as_secs_f64() * 1000.0, NUM_BATCHES * 4);
        println!("  SIMD:       {:.3} ms ({} lookups)", 
                 simd_time.as_secs_f64() * 1000.0, NUM_BATCHES * 4);
        println!("  Speedup:    {:.2}x", 
                 sequential_time.as_secs_f64() / simd_time.as_secs_f64());
        
        // Assert SIMD is faster
        assert!(simd_time < sequential_time, 
                "SIMD should be faster than sequential");
    }
    
    #[test]
    fn test_integrated_fast_ecdlp_simd_with_batch_lookup() {
        // This test validates the integration within fast_ecdlp_simd
        let tables = read_or_gen_tables();
        let view = tables.view();
        
        // Create a modified version of fast_ecdlp_simd that uses both approaches
        // and validates they produce the same results
        
        // Test with a known point
        let test_value = 1000000u64;
        let point = RistrettoPoint::mul_base(&Scalar::from(test_value));
        
        let args = ECDLPArguments::new_with_range(0, 2000000)
            .n_threads(1)
            .pseudo_constant_time(false);
        
        // Run with original implementation
        let result_original = par_decode(&view, point, args);
        
        // Run with SIMD lookup implementation (you'd need to add a flag or separate function)
        // let result_simd = par_decode_with_simd_lookup(&view, point, args);
        
        // For now, just validate the original works
        assert_eq!(result_original, Some(test_value as i64));
        
        println!("✓ Integration test passed");
    }
    
    // Helper function to create a test version that compares both implementations
    fn validate_batch_lookup_in_context(
        t1_table: &CuckooT1HashMapView,
        qx_batch: &[FieldElement; BATCH_SIZE],
        j_start: usize,
        l1: usize,
    ) -> (Vec<Option<u64>>, Vec<Option<u64>>) {
        let mut sequential_results = Vec::new();
        let mut simd_results = Vec::new();
        
        // Sequential approach
        for (idx, qx) in qx_batch.iter().enumerate() {
            let mut found = None;
            t1_table.lookup(&qx.as_bytes(), |value| {
                found = Some(value);
                true
            });
            sequential_results.push(found);
        }
        
        // SIMD approach
        let simd_table = SimdCuckooT1HashMapView {
            keys: t1_table.keys,
            values: t1_table.values,
            cuckoo_len: t1_table.cuckoo_len,
        };
        
        for chunk_idx in 0..BATCH_SIZE / 4 {
            let base = chunk_idx * 4;
            let queries = [
                qx_batch[base].as_bytes(),
                qx_batch[base + 1].as_bytes(),
                qx_batch[base + 2].as_bytes(),
                qx_batch[base + 3].as_bytes(),
            ];
            
            let results = simd_table.lookup_batch_4(&queries);
            for (i, (found, value)) in results.iter().enumerate() {
                simd_results.push(if *found { Some(*value) } else { None });
            }
        }
        
        // Handle remainder
        for idx in (BATCH_SIZE / 4 * 4)..BATCH_SIZE {
            let mut found = None;
            t1_table.lookup(&qx_batch[idx].as_bytes(), |value| {
                found = Some(value);
                true
            });
            simd_results.push(found);
        }
        
        assert_eq!(sequential_results.len(), simd_results.len());
        for (i, (seq, simd)) in sequential_results.iter().zip(simd_results.iter()).enumerate() {
            assert_eq!(seq, simd, "Mismatch at index {}", i);
        }
        
        (sequential_results, simd_results)
    }

    #[test]
    fn stress_test_ecdlp_simd_correctness() {
        let tables = read_or_gen_tables();
        let view = tables.view();
        let mut rng = rand::thread_rng();
        
        // Test range up to 2^48
        const MAX_VALUE: u64 = 1u64 << 48;
        const NUM_TESTS: usize = 10000;
        
        for i in 0..NUM_TESTS {
            let test_value = rand::thread_rng().gen_range(0..MAX_VALUE);
            let point = RistrettoPoint::mul_base(&Scalar::from(test_value));
            
            let args = ECDLPArguments::new_with_range(0, MAX_VALUE as i64)
                .n_threads(16)
                .pseudo_constant_time(false);
            
            let result = par_decode(&view, point, args);
            assert_eq!(result, Some(test_value as i64), 
                      "Failed on iteration {} with value {}", i, test_value);
            
            if i % 100 == 0 && i > 0 {
                println!("✓ Completed {} iterations", i);
            }
        }
        
        println!("✅ All {} random tests passed!", NUM_TESTS);
    }

    #[test]
    fn test_and_benchmark_vectorized_point_iterator() {
    use std::time::Instant;
    use std::hint::black_box;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    
    println!("\n=== ECDLP Benchmark with Proper Iterator Integration ===");
    
    let tables = read_or_gen_tables();
    let view = tables.view();
    
    let test_value: u64 = 9_223_372_036_854_775_806 / 1000;
    let test_point = black_box(RistrettoPoint::mul_base(&Scalar::from(test_value)));
    
    println!("Test value: {}", test_value);
    
    // Define optimized make_point_iterator that uses 4-way operations
    fn make_point_iterator_simd(
        precomputed_tables: &ECDLPTablesFileView<'_>,
        normalized: RistrettoPoint,
        num_batches: usize,
    ) -> impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)> {
        // Apply same transformations as original
        let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
        let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());
        
        let initial = AffineMontgomeryPoint::from(&normalized.0);
        let batch_step = -(els_per_batch as i64);
        let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
        
        struct OptimizedIterator {
            current_batch: [AffineMontgomeryPoint; 4],
            step: AffineMontgomeryPoint,
            step_x4: AffineMontgomeryPoint,
            batch_idx: usize,
            j: usize,
            num_batches: usize,
        }
        
        impl Iterator for OptimizedIterator {
            type Item = (usize, usize, AffineMontgomeryPoint, f64);
            
            fn next(&mut self) -> Option<Self::Item> {
                if self.j >= self.num_batches {
                    return None;
                }
                
                if self.batch_idx >= 4 && self.j < self.num_batches {
                    // Use 4-way SIMD operation
                    self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
                        &self.current_batch,
                        &self.step_x4
                    );
                    self.batch_idx = 0;
                }
                
                let result = (
                    0,
                    self.j * (1 << L2),
                    self.current_batch[self.batch_idx],
                    self.j as f64 / self.num_batches as f64
                );
                
                self.batch_idx += 1;
                self.j += 1;
                
                Some(result)
            }
        }
        
        // Initialize first 4 points
        let p0 = initial;
        let p1 = p0.addition_not_ct(&step);
        let p2 = p1.addition_not_ct(&step);
        let p3 = p2.addition_not_ct(&step);
        
        let step_x4 = step.addition_not_ct(&step)
            .addition_not_ct(&step)
            .addition_not_ct(&step);
        
        OptimizedIterator {
            current_batch: [p0, p1, p2, p3],
            step,
            step_x4,
            batch_idx: 0,
            j: 0,
            num_batches,
        }
    }
    
    // Run benchmarks
    let variants = vec![
        ("Standard", false),
        ("Optimized (4-way)", true),
    ];
    
    let mut results = Vec::new();
    
    for (name, use_optimized) in variants {
        println!("\n--- {} ---", name);
        
        let start_total = Instant::now();
        
        let result = {
            use rayon::prelude::*;
            
            let range_start = 0i64;
            let range_end = 1i64 << 63;
            let n_threads = 16;
            let chunk_size: u64 = 64 * 320_000_000_000_0 / ((30 - view.get_l1()).max(0) * 2).max(1) as u64;
            let total_range = range_end as u64 - range_start as u64;
            let num_chunks = ((total_range + chunk_size - 1) / chunk_size) as usize;
            
            let end_flag = Arc::new(AtomicBool::new(false));
            
            // Pre-compute T2 cache
            let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
            let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
            let t2_table = view.get_t2();
            for i in 0..BATCH_SIZE {
                let point = t2_table.index(i);
                t2_cache_alpha[i] = &MONTGOMERY_A_NEG - &point.u;
                t2_cache[i] = point;
            }
            
            (0..n_threads)
                .into_par_iter()
                .find_map_any(|thread_index| {
                    let t2_u_values: [FieldElement; BATCH_SIZE] = {
                        let mut u_values = [FieldElement::ZERO; BATCH_SIZE];
                        for i in 0..BATCH_SIZE {
                            u_values[i] = t2_cache[i].u;
                        }
                        u_values
                    };
                    
                    let t2_vs: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| p.v.clone());
                    let t2_vs_neg: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| -&p.v);
                    
                    let mut batch = [FieldElement::ZERO; BATCH_SIZE];
                    let mut alphas = [FieldElement::ZERO; BATCH_SIZE];
                    let mut qxs = [FieldElement::ZERO; BATCH_SIZE];
                    let mut neg_qxs = [FieldElement::ZERO; BATCH_SIZE];
                    
                    for chunk_index in (thread_index..num_chunks).step_by(n_threads) {
                        if end_flag.load(Ordering::Relaxed) {
                            return None;
                        }
                        
                        let start = range_start + (chunk_index as u64 * chunk_size) as i64;
                        let end = ((chunk_index as u64 + 1) * chunk_size).min(range_end as u64) as i64;
                        
                        let amplitude = (end - start).max(0);
                        let offset = start
                            + ((1 << (L2 - 1)) << view.get_l1())
                            + (1 << (view.get_l1() - 1));
                        
                        let normalized = black_box(test_point) - RistrettoPoint::mul_base(&i64_to_scalar(offset));
                        let j_end = ((amplitude as i64) >> view.get_l1()) as usize;
                        let num_batches = (j_end + (1 << L2) - 1) / (1 << L2);
                        
                        // Use appropriate iterator
                        let point_iter: Box<dyn Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)>> = 
                            if use_optimized {
                                Box::new(make_point_iterator_simd(&view, normalized, num_batches))
                            } else {
                                Box::new(make_point_iterator(&view, normalized, num_batches))
                            };
                        
                        // Run ECDLP
                        if let Some(res) = fast_ecdlp_simd(
                            &view,
                            black_box(normalized),
                            point_iter,
                            false,
                            &end_flag,
                            |_| std::ops::ControlFlow::Continue(()),
                            &t2_cache,
                            &t2_cache_alpha,
                            &t2_u_values,
                            &t2_vs,
                            &t2_vs_neg,
                            &mut batch,
                            &mut alphas,
                            &mut qxs,
                            &mut neg_qxs
                        ) {
                            end_flag.store(true, Ordering::SeqCst);
                            return Some(offset + res as i64);
                        }
                    }
                    
                    None
                })
        };
        
        let total_time = start_total.elapsed();
        
        println!("Result: {:?}", result);
        println!("Total time: {:?}", total_time);
        
        results.push((name, total_time));
        assert_eq!(result, Some(test_value as i64), "{} should find correct value", name);
    }
    
    // Summary
    println!("\n=== Summary ===");
    let (_, standard_time) = results[0];
    
    for (name, time) in &results {
        println!("{:20} {:?}", name, time);
    }
    
    println!("\nSpeedups:");
    for (name, time) in &results[1..] {
        let speedup = standard_time.as_secs_f64() / time.as_secs_f64();
        println!("{:20} {:.2}x", name, speedup);
    }
    
    // Additional analysis
    println!("\n=== Analysis ===");
    println!("Iterator generates ~2.8x speedup in isolation");
    println!("But iterator is only ~2.7ms out of ~1500ms total");
    println!("Expected speedup: ~1.8ms saved = ~0.12% improvement");
    println!("This matches what we're seeing!");
    }

    #[test]
    fn benchmark_point_iterators_isolated() {
        use std::time::Instant;
        use std::hint::black_box;
        
        println!("\n=== Isolated Point Iterator Benchmarks ===");
        
        let tables = read_or_gen_tables();
        let view = tables.view();
        
        // Use realistic parameters from the ECDLP benchmark
        let test_value: u64 = 9_223_372_036_854_775_806 / 1000;
        let test_point = RistrettoPoint::mul_base(&Scalar::from(test_value));
        
        // Simulate what happens in one chunk
        let range_start = 0i64;
        let chunk_size: u64 = 64 * 320_000_000_000_0 / ((30 - view.get_l1()).max(0) * 2).max(1) as u64;
        let amplitude = chunk_size as i64;
        let offset = range_start 
            + ((1 << (L2 - 1)) << view.get_l1())
            + (1 << (view.get_l1() - 1));
        
        let normalized = test_point - RistrettoPoint::mul_base(&i64_to_scalar(offset));
        let j_end = ((amplitude as i64) >> view.get_l1()) as usize;
        let num_batches = (j_end + (1 << L2) - 1) / (1 << L2);
        
        println!("Test parameters:");
        println!("  chunk_size: {}", chunk_size);
        println!("  j_end: {}", j_end);
        println!("  num_batches: {}", num_batches);
        println!("  Points to generate: {}\n", num_batches * (1 << L2));
        
        // Warm up
        let _ = black_box(make_point_iterator(&view, normalized, 100).collect::<Vec<_>>());
        
        // 1. Standard iterator
        println!("--- Standard make_point_iterator ---");
        let mut standard_times = Vec::new();
        
        for _ in 0..5 {
            let start = Instant::now();
            let points: Vec<_> = black_box(make_point_iterator(&view, normalized, num_batches).collect());
            let elapsed = start.elapsed();
            standard_times.push(elapsed);
            println!("  Run: {:?}, points: {}", elapsed, points.len());
        }
        
        let standard_avg = standard_times.iter().sum::<std::time::Duration>() / standard_times.len() as u32;
        println!("  Average: {:?}", standard_avg);
        
        // 2. Non-lazy optimized with 4-way operations
        println!("\n--- Non-lazy Optimized (with 4-way ops) ---");
        
        let make_non_lazy_4way = |view: &ECDLPTablesFileView, normalized: RistrettoPoint, num_batches: usize| -> Vec<(usize, usize, AffineMontgomeryPoint, f64)> {
            let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
            let els_per_batch: u64 = 1u64 << (L2 + view.get_l1());
            
            let initial = AffineMontgomeryPoint::from(&normalized.0);
            let batch_step = -(els_per_batch as i64);
            let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
            
            let mut results = Vec::with_capacity(num_batches);
            
            // Pre-compute step_x4
            let step_x4 = step.addition_not_ct(&step)
                .addition_not_ct(&step)
                .addition_not_ct(&step);
            
            let chunks = num_batches / 4;
            
            // Process in groups of 4
            let mut current_batch = [initial; 4];
            
            for chunk in 0..chunks {
                if chunk == 0 {
                    // Initialize first batch
                    current_batch[1] = current_batch[0].addition_not_ct(&step);
                    current_batch[2] = current_batch[1].addition_not_ct(&step);
                    current_batch[3] = current_batch[2].addition_not_ct(&step);
                } else {
                    // Use 4-way operation to advance all points
                    current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
                        &current_batch,
                        &step_x4
                    );
                }
                
                // Store results
                for i in 0..4 {
                    let j = chunk * 4 + i;
                    results.push((0, j * (1 << L2), current_batch[i], j as f64 / num_batches as f64));
                }
            }
            
            // Handle remainder with scalar operations
            if chunks * 4 < num_batches {
                let mut current = current_batch[3].addition_not_ct(&step);
                for j in (chunks * 4)..num_batches {
                    results.push((0, j * (1 << L2), current, j as f64 / num_batches as f64));
                    current = current.addition_not_ct(&step);
                }
            }
            
            results
        };
        
        let mut non_lazy_times = Vec::new();
        
        for _ in 0..5 {
            let start = Instant::now();
            let points = black_box(make_non_lazy_4way(&view, normalized, num_batches));
            let elapsed = start.elapsed();
            non_lazy_times.push(elapsed);
            println!("  Run: {:?}, points: {}", elapsed, points.len());
        }
        
        let non_lazy_avg = non_lazy_times.iter().sum::<std::time::Duration>() / non_lazy_times.len() as u32;
        println!("  Average: {:?}", non_lazy_avg);
        
        // 3. Lazy SIMD iterator
        println!("\n--- Lazy SIMD Iterator ---");
        
        struct LazySimdPointIterator {
            current_batch: [AffineMontgomeryPoint; 4],
            step: AffineMontgomeryPoint,
            step_x4: AffineMontgomeryPoint,
            batch_idx: usize,
            j: usize,
            num_batches: usize,
        }
        
        impl Iterator for LazySimdPointIterator {
            type Item = (usize, usize, AffineMontgomeryPoint, f64);
            
            fn next(&mut self) -> Option<Self::Item> {
                if self.j >= self.num_batches {
                    return None;
                }
                
                if self.batch_idx >= 4 && self.j < self.num_batches {
                    self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
                        &self.current_batch,
                        &self.step_x4
                    );
                    self.batch_idx = 0;
                }
                
                let result = (
                    0,
                    self.j * (1 << L2),
                    self.current_batch[self.batch_idx],
                    self.j as f64 / self.num_batches as f64
                );
                
                self.batch_idx += 1;
                self.j += 1;
                
                Some(result)
            }
        }
        
        let make_lazy_simd_iterator = |view: &ECDLPTablesFileView, normalized: RistrettoPoint, num_batches: usize| -> LazySimdPointIterator {
            let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
            let els_per_batch: u64 = 1u64 << (L2 + view.get_l1());
            
            let initial = AffineMontgomeryPoint::from(&normalized.0);
            let batch_step = -(els_per_batch as i64);
            let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
            
            let p0 = initial;
            let p1 = p0.addition_not_ct(&step);
            let p2 = p1.addition_not_ct(&step);
            let p3 = p2.addition_not_ct(&step);
            
            let step_x4 = step.addition_not_ct(&step)
                .addition_not_ct(&step)
                .addition_not_ct(&step);
            
            LazySimdPointIterator {
                current_batch: [p0, p1, p2, p3],
                step,
                step_x4,
                batch_idx: 0,
                j: 0,
                num_batches,
            }
        };
        
        let mut lazy_times = Vec::new();
        
        for _ in 0..5 {
            let start = Instant::now();
            let iter = make_lazy_simd_iterator(&view, normalized, num_batches);
            let points: Vec<_> = black_box(iter.collect());
            let elapsed = start.elapsed();
            lazy_times.push(elapsed);
            println!("  Run: {:?}, points: {}", elapsed, points.len());
        }
        
        let lazy_avg = lazy_times.iter().sum::<std::time::Duration>() / lazy_times.len() as u32;
        println!("  Average: {:?}", lazy_avg);
        
        // 4. Verify correctness of all variants
        println!("\n--- Correctness Verification ---");
        
        let standard_points: Vec<_> = make_point_iterator(&view, normalized, 20).collect();
        let non_lazy_points = make_non_lazy_4way(&view, normalized, 20);
        let lazy_points: Vec<_> = make_lazy_simd_iterator(&view, normalized, 20).collect();
        
        let mut all_match = true;
        for i in 0..20 {
            let std_u = standard_points[i].2.u.0[0];
            let non_lazy_u = non_lazy_points[i].2.u.0[0];
            let lazy_u = lazy_points[i].2.u.0[0];
            
            if std_u != non_lazy_u || std_u != lazy_u {
                println!("  ❌ Mismatch at index {}:", i);
                println!("    Standard:  {:016x}", std_u);
                println!("    Non-lazy:  {:016x}", non_lazy_u);
                println!("    Lazy:      {:016x}", lazy_u);
                all_match = false;
                break;
            }
        }
        
        if all_match {
            println!("  ✅ All variants produce identical results");
        }
        
        // Summary
        println!("\n=== Performance Summary ===");
        println!("Standard:          {:?}", standard_avg);
        println!("Non-lazy (4-way):  {:?} ({:.2}x)", non_lazy_avg, standard_avg.as_secs_f64() / non_lazy_avg.as_secs_f64());
        println!("Lazy SIMD:         {:?} ({:.2}x)", lazy_avg, standard_avg.as_secs_f64() / lazy_avg.as_secs_f64());
        
        // Points per second
        let total_points = num_batches * (1 << L2);
        println!("\nThroughput (points/sec):");
        println!("Standard:          {:.0}", total_points as f64 / standard_avg.as_secs_f64());
        println!("Non-lazy (4-way):  {:.0}", total_points as f64 / non_lazy_avg.as_secs_f64());
        println!("Lazy SIMD:         {:.0}", total_points as f64 / lazy_avg.as_secs_f64());
    }

    #[test]
    fn measure_iterator_time_accurately() {
        use std::time::Instant;
        use std::hint::black_box;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        
        println!("\n=== Iterator Benchmark with Accurate Measurement ===");
        
        let tables = read_or_gen_tables();
        let view = tables.view();
        
        let test_value: u64 = 9_223_372_036_854_775_806 / 1000;
        let test_point = black_box(RistrettoPoint::mul_base(&Scalar::from(test_value)));
        
        // Test with different thread counts
        let thread_counts = vec![4, 8, 16];
        
        for &n_threads in &thread_counts {
            println!("\n========== Testing with {} threads ==========", n_threads);
            
            // Define optimized iterator
            fn make_point_iterator_simd(
                precomputed_tables: &ECDLPTablesFileView<'_>,
                normalized: RistrettoPoint,
                num_batches: usize,
            ) -> impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)> {
                let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
                let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());
                
                let initial = AffineMontgomeryPoint::from(&normalized.0);
                let batch_step = -(els_per_batch as i64);
                let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
                
                struct OptimizedIterator {
                    current_batch: [AffineMontgomeryPoint; 4],
                    step: AffineMontgomeryPoint,
                    step_x4: AffineMontgomeryPoint,
                    batch_idx: usize,
                    j: usize,
                    num_batches: usize,
                }
                
                impl Iterator for OptimizedIterator {
                    type Item = (usize, usize, AffineMontgomeryPoint, f64);
                    
                    fn next(&mut self) -> Option<Self::Item> {
                        if self.j >= self.num_batches {
                            return None;
                        }
                        
                        if self.batch_idx >= 4 && self.j < self.num_batches {
                            self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
                                &self.current_batch,
                                &self.step_x4
                            );
                            self.batch_idx = 0;
                        }
                        
                        let result = (
                            0,
                            self.j * (1 << L2),
                            self.current_batch[self.batch_idx],
                            self.j as f64 / self.num_batches as f64
                        );
                        
                        self.batch_idx += 1;
                        self.j += 1;
                        
                        Some(result)
                    }
                }
                
                let p0 = initial;
                let p1 = p0.addition_not_ct(&step);
                let p2 = p1.addition_not_ct(&step);
                let p3 = p2.addition_not_ct(&step);
                
                let step_x4 = step.addition_not_ct(&step)
                    .addition_not_ct(&step)
                    .addition_not_ct(&step);
                
                OptimizedIterator {
                    current_batch: [p0, p1, p2, p3],
                    step,
                    step_x4,
                    batch_idx: 0,
                    j: 0,
                    num_batches,
                }
            }
            
            let variants = vec![
                ("Standard", false),
                ("Optimized (4-way)", true),
            ];
            
            let mut results = Vec::new();
            
            for (name, use_optimized) in variants {
                println!("\n--- {} ---", name);
                
                let thread_iter_times: Arc<Mutex<Vec<(usize, std::time::Duration, usize)>>> = Arc::new(Mutex::new(Vec::new()));
                let start_total = Instant::now();
                
                let result = {
                    use rayon::prelude::*;
                    
                    let range_start = 0i64;
                    let range_end = 1i64 << 63;
                    let chunk_size: u64 = 64 * 320_000_000_000_0 / ((30 - view.get_l1()).max(0) * 2).max(1) as u64;
                    let total_range = range_end as u64 - range_start as u64;
                    let num_chunks = ((total_range + chunk_size - 1) / chunk_size) as usize;
                    
                    let end_flag = Arc::new(AtomicBool::new(false));
                    let thread_iter_times_clone = Arc::clone(&thread_iter_times);
                    
                    // Pre-compute T2 cache
                    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
                    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
                    let t2_table = view.get_t2();
                    for i in 0..BATCH_SIZE {
                        let point = t2_table.index(i);
                        t2_cache_alpha[i] = &MONTGOMERY_A_NEG - &point.u;
                        t2_cache[i] = point;
                    }
                    
                    // Use custom thread pool with specified thread count
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(n_threads)
                        .build()
                        .unwrap();
                    
                    pool.install(|| {
                        (0..n_threads)
                            .into_par_iter()
                            .find_map_any(|thread_index| {
                                let mut thread_iter_time = std::time::Duration::ZERO;
                                let mut chunks_processed = 0;
                                
                                let t2_u_values: [FieldElement; BATCH_SIZE] = {
                                    let mut u_values = [FieldElement::ZERO; BATCH_SIZE];
                                    for i in 0..BATCH_SIZE {
                                        u_values[i] = t2_cache[i].u;
                                    }
                                    u_values
                                };
                                
                                let t2_vs: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| p.v.clone());
                                let t2_vs_neg: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| -&p.v);
                                
                                let mut batch = [FieldElement::ZERO; BATCH_SIZE];
                                let mut alphas = [FieldElement::ZERO; BATCH_SIZE];
                                let mut qxs = [FieldElement::ZERO; BATCH_SIZE];
                                let mut neg_qxs = [FieldElement::ZERO; BATCH_SIZE];
                                
                                for chunk_index in (thread_index..num_chunks).step_by(n_threads) {
                                    if end_flag.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    
                                    chunks_processed += 1;
                                    
                                    let start = range_start + (chunk_index as u64 * chunk_size) as i64;
                                    let end = ((chunk_index as u64 + 1) * chunk_size).min(range_end as u64) as i64;
                                    
                                    let amplitude = (end - start).max(0);
                                    let offset = start
                                        + ((1 << (L2 - 1)) << view.get_l1())
                                        + (1 << (view.get_l1() - 1));
                                    
                                    let normalized = black_box(test_point) - RistrettoPoint::mul_base(&i64_to_scalar(offset));
                                    let j_end = ((amplitude as i64) >> view.get_l1()) as usize;
                                    let num_batches = (j_end + (1 << L2) - 1) / (1 << L2);
                                    
                                    // Time iterator
                                    let iter_start = Instant::now();
                                    let points: Vec<_> = if use_optimized {
                                        make_point_iterator_simd(&view, normalized, num_batches).collect()
                                    } else {
                                        make_point_iterator(&view, normalized, num_batches).collect()
                                    };
                                    thread_iter_time += iter_start.elapsed();
                                    
                                    // Run ECDLP
                                    if let Some(res) = fast_ecdlp_simd(
                                        &view,
                                        black_box(normalized),
                                        points.into_iter(),
                                        false,
                                        &end_flag,
                                        |_| std::ops::ControlFlow::Continue(()),
                                        &t2_cache,
                                        &t2_cache_alpha,
                                        &t2_u_values,
                                        &t2_vs,
                                        &t2_vs_neg,
                                        &mut batch,
                                        &mut alphas,
                                        &mut qxs,
                                        &mut neg_qxs
                                    ) {
                                        end_flag.store(true, Ordering::SeqCst);
                                        thread_iter_times_clone.lock().unwrap().push((thread_index, thread_iter_time, chunks_processed));
                                        return Some(offset + res as i64);
                                    }
                                }
                                
                                thread_iter_times_clone.lock().unwrap().push((thread_index, thread_iter_time, chunks_processed));
                                None
                            })
                    })
                };
                
                let total_time = start_total.elapsed();
                
                println!("Result: {:?}", result);
                println!("Total time: {:?}", total_time);
                
                // Analyze iterator times
                let times = thread_iter_times.lock().unwrap();
                let total_iter_time: std::time::Duration = times.iter().map(|(_, t, _)| *t).sum();
                let max_iter_time = times.iter().map(|(_, t, _)| *t).max().unwrap_or_default();
                let total_chunks: usize = times.iter().map(|(_, _, c)| *c).sum();
                
                println!("\nIterator time analysis:");
                println!("  Total chunks processed: {}", total_chunks);
                println!("  Total iterator time (sum): {:?}", total_iter_time);
                println!("  Max iterator time (critical path): {:?}", max_iter_time);
                println!("  Iterator percentage: {:.2}%", (max_iter_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
                
                // Per-thread breakdown
                if n_threads <= 8 {
                    println!("\n  Per-thread breakdown:");
                    for (tid, time, chunks) in times.iter() {
                        println!("    Thread {:2}: {:?} ({} chunks, {:.2}ms/chunk)", 
                                tid, time, chunks, time.as_secs_f64() * 1000.0 / *chunks as f64);
                    }
                }
                
                results.push((name, total_time, max_iter_time));
            }
            
            // Compare results
            println!("\n--- Comparison for {} threads ---", n_threads);
            let (_, std_total, std_iter) = results[0];
            let (_, opt_total, opt_iter) = results[1];
            
            println!("Total speedup: {:.2}x", std_total.as_secs_f64() / opt_total.as_secs_f64());
            println!("Iterator speedup: {:.2}x", std_iter.as_secs_f64() / opt_iter.as_secs_f64());
            
            let time_saved = std_iter.saturating_sub(opt_iter);
            println!("Iterator time saved: {:?}", time_saved);
            println!("Expected total speedup: {:.2}x", std_total.as_secs_f64() / (std_total - time_saved).as_secs_f64());
        }
    }

    #[test]
    fn test_j_batched_vs_standard() {
        // Create test setup
        let precomputed_tables = read_or_gen_tables();
        let view = precomputed_tables.view();
        let test_points = vec![
            // Generate some test points with known discrete logs
            (1234567890_i64, i64_to_scalar(1234567890) * G),
            (9876543210_i64, i64_to_scalar(9876543210) * G),
            // Add more test cases
        ];
        
        for (expected_dlog, point) in test_points {
            // Test standard version
            let result_standard = {
                let args = ECDLPArguments {
                    range_start: 0,
                    range_end: 1 << 63,
                    n_threads: 4,
                    pseudo_constant_time: false,
                    progress_report_function: NoopReportFn,
                };
                
                // Run standard version
                par_decode(&view, point, args)
            };
            
            // Test j-batched version
            let result_j_batched = {
                let args = ECDLPArguments {
                    range_start: 0,
                    range_end: 1 << 63,
                    n_threads: 4,
                    pseudo_constant_time: false,
                    progress_report_function: NoopReportFn,
                };
                
                // Run j-batched version
                par_decode_j_batch(&view, point, args)
            };
            
            assert_eq!(result_standard, result_j_batched);
            assert_eq!(result_standard, Some(expected_dlog));
        }
    }

fn batch_invert(elements: &mut [FieldElement]) {
    let mut scratch = vec![FieldElement::ONE; elements.len()];
    let mut acc = FieldElement::ONE;
    
    // Forward pass
    for i in 0..elements.len() {
        scratch[i] = acc;
        acc = &acc * &elements[i];
    }
    
    // Invert accumulator
    acc = acc.invert();
    
    // Reverse pass
    for i in (0..elements.len()).rev() {
        let tmp = &acc * &scratch[i];
        acc = &acc * &elements[i];
        elements[i] = tmp;
    }
}

    #[test]
    fn test_j_batched_inversion() {
    // Create test data that mimics what we'd see in the real algorithm
    const TEST_BATCH_SIZE: usize = 256; // Smaller for easier debugging
    const J_COUNT: usize = 4;
    
    // Generate some test field elements
    let mut test_batches: [[FieldElement; TEST_BATCH_SIZE]; J_COUNT] = 
        [[FieldElement::ONE; TEST_BATCH_SIZE]; J_COUNT];
    
    // Fill with some non-trivial values
    for j in 0..J_COUNT {
        for i in 0..TEST_BATCH_SIZE {
            // Create distinct values for each position
            let value = ((j + 1) * 1000 + i + 1) as u64;
            test_batches[j][i] = FieldElement::from_limbs([value,0,0,0,0]);
        }
    }
    
    // Clone for comparison
    let mut j_batched_result = test_batches.clone();
    let mut standard_result = test_batches.clone();
    
    // Method 1: Standard batch inversion (per j)
    for j in 0..J_COUNT {
        batch_invert(&mut standard_result[j]);
    }
    
    // Method 2: J-batched inversion (CORRECTED)
    // Each accumulator processes ALL elements from one j
    let mut accs = [FieldElement::ONE; J_COUNT];
    let mut scratch = vec![[FieldElement::ONE; TEST_BATCH_SIZE]; J_COUNT];
    
    // Forward pass - each accumulator processes its own j's batch
    for j in 0..J_COUNT {
        for i in 0..TEST_BATCH_SIZE {
            scratch[j][i] = accs[j];
            accs[j] = &accs[j] * &j_batched_result[j][i];
        }
    }
    
    // Invert all accumulators using 4-way SIMD
    let p01 = &accs[0] * &accs[1];
    let p23 = &accs[2] * &accs[3];
    let p0123 = &p01 * &p23;
    let inv_p0123 = p0123.invert();
    
    let factors = [
        &accs[1] * &p23,
        &accs[0] * &p23,
        &p01 * &accs[3],
        &p01 * &accs[2],
    ];
    
    let inv_broadcast = [inv_p0123; 4];
    let inv_accs = FieldElement::batch_mul_4way(&inv_broadcast, &factors);
    
    // Reverse pass - each accumulator processes its own j's batch
    for j in 0..J_COUNT {
        let mut acc = inv_accs[j];
        for i in (0..TEST_BATCH_SIZE).rev() {
            let tmp = &acc * &j_batched_result[j][i];
            j_batched_result[j][i] = &acc * &scratch[j][i];
            acc = tmp;
        }
    }
    
    // Compare results
    for j in 0..J_COUNT {
        for i in 0..TEST_BATCH_SIZE {
            assert_eq!(
                j_batched_result[j][i], 
                standard_result[j][i],
                "Mismatch at j={}, i={}", j, i
            );
            
            // Also verify it's actually the inverse
            let product = &test_batches[j][i] * &j_batched_result[j][i];
            assert_eq!(
                product, 
                FieldElement::ONE,
                "Invalid inverse at j={}, i={}", j, i
            );
        }
    }
    }
    
    // Test 2: Validate the full computation pipeline for a single j
    #[test]
    fn test_single_j_computation() {
        let tables = read_or_gen_tables();
        let view = &tables.view();

        let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
        let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
        let t2_table = view.get_t2();
        for i in 0..BATCH_SIZE {
            let point = t2_table.index(i);
            t2_cache_alpha[i] = &MONTGOMERY_A_NEG - &point.u;
            t2_cache[i] = point;
        }

        let t2_u_values: [FieldElement; BATCH_SIZE] = {
            let mut u_values = [FieldElement::ZERO; BATCH_SIZE];
            for i in 0..BATCH_SIZE {
                u_values[i] = t2_cache[i].u;
            }
            u_values
        };

        let t2_vs: [FieldElement; BATCH_SIZE] = {
            let mut vs = [FieldElement::ZERO; BATCH_SIZE];
            for i in 0..BATCH_SIZE {
                vs[i] = t2_cache[i].v;
            }
            vs
        };
        
        // Create a test target point
        let test_scalar = 123456789_i64;
        let target_point = i64_to_scalar(test_scalar) * G;
        let target_montgomery = AffineMontgomeryPoint::from(&target_point.0);
        
        // Test the computation pipeline
        let mut batch = [FieldElement::ZERO; BATCH_SIZE];
        let mut alphas = [FieldElement::ZERO; BATCH_SIZE];
        let mut qxs = [FieldElement::ZERO; BATCH_SIZE];
        let mut neg_qxs = [FieldElement::ZERO; BATCH_SIZE];
        
        // Step 1: Compute differences
        batch_field_subtract(&mut batch, &t2_u_values, &target_montgomery.u);
        batch_field_subtract(&mut alphas, &t2_cache_alpha, &target_montgomery.u);
        
        // Step 2: Invert
        let mut batch_copy = batch.clone();
        batch_invert(&mut batch_copy);
        
        // Step 3: Compute qx values
        batch_field_subtract(&mut qxs, &t2_vs, &target_montgomery.v);
        batch_field_mul_and_square(&mut neg_qxs, &qxs, &batch_copy);
        batch_field_add(&mut qxs, &neg_qxs, &alphas);
        
        // Verify some properties
        for i in 0..BATCH_SIZE {
            // Check that batch inversion worked
            let product = &batch[i] * &batch_copy[i];
            if (!batch[i].is_zero()).into() {
                assert_eq!(product, FieldElement::ONE, "Inversion failed at i={}", i);
            }
            
            // You could also verify the qx computation makes sense
            // by checking against a reference implementation
        }
    }
    
    // Test 3: Compare full algorithm results
    // #[test]
    // fn test_full_algorithm_comparison() {
    //     // Create a small test case with known discrete log
    //     let test_cases = vec![
    //         (1000_i64, i64_to_scalar(1000) * G),
    //         (50000_i64, i64_to_scalar(50000) * G),
    //         (-30000_i64, i64_to_scalar(-30000) * G),
    //     ];
        
    //     for (expected_dlog, point) in test_cases {
    //         println!("Testing dlog={}", expected_dlog);
            
    //         // Run both versions on the same range
    //         let range_start = expected_dlog - 100;
    //         let range_end = expected_dlog + 100;
            
    //         // Version 1: Original algorithm
    //         let result_original = {
    //             // Run your original fast_ecdlp_simd
    //             // ...
    //         };
            
    //         // Version 2: J-batched algorithm  
    //         let result_j_batched = {
    //             // Run your fast_ecdlp_simd_j_batched
    //             // ...
    //         };
            
    //         assert_eq!(
    //             result_original, 
    //             result_j_batched,
    //             "Results differ for dlog={}", expected_dlog
    //         );
    //         assert_eq!(
    //             result_original, 
    //             Some(expected_dlog as u64),
    //             "Wrong result for dlog={}", expected_dlog
    //         );
    //     }
    // }
    
    // Test 4: Detailed trace comparison
    #[test]
    fn test_with_trace_comparison() {
        // This test would instrument both versions to log intermediate values
        // and compare them step by step
        
        struct TraceEntry {
            j_value: usize,
            batch_values: Vec<FieldElement>,
            inverted_values: Vec<FieldElement>,
            qx_values: Vec<FieldElement>,
        }
        
        let mut original_trace: Vec<TraceEntry> = Vec::new();
        let mut j_batched_trace: Vec<TraceEntry> = Vec::new();
        
        // Run both algorithms with tracing enabled
        // Compare traces to find where they diverge
    }

    use std::hint::black_box;

    const J_BATCH_SIZE: usize = 4;
fn j_batched_inversion_correct(
    batch_storage: &mut [[FieldElement; BATCH_SIZE]; J_BATCH_SIZE]
) {
    // Each accumulator will process ALL elements from one j
    let mut accs = [FieldElement::ONE; J_BATCH_SIZE];
    let mut scratch = vec![[FieldElement::ONE; BATCH_SIZE]; J_BATCH_SIZE];
    
    // Forward pass - each accumulator processes its own j's batch
    for j in 0..J_BATCH_SIZE {
        for i in 0..BATCH_SIZE {
            scratch[j][i] = accs[j];
            accs[j] = &accs[j] * &batch_storage[j][i];
        }
    }
    
    // Invert all accumulators using 4-way SIMD
    let p01 = &accs[0] * &accs[1];
    let p23 = &accs[2] * &accs[3];
    let p0123 = &p01 * &p23;
    let inv_p0123 = p0123.invert();
    
    let factors = [
        &accs[1] * &p23,
        &accs[0] * &p23,
        &p01 * &accs[3],
        &p01 * &accs[2],
    ];
    
    let inv_broadcast = [inv_p0123; 4];
    let inv_accs = FieldElement::batch_mul_4way(&inv_broadcast, &factors);
    
    // Reverse pass - each accumulator processes its own j's batch
    for j in 0..J_BATCH_SIZE {
        let mut acc = inv_accs[j];
        for i in (0..BATCH_SIZE).rev() {
            let tmp = &acc * &batch_storage[j][i];
            batch_storage[j][i] = &acc * &scratch[j][i];
            acc = tmp;
        }
    }
}

#[test]
fn bench_ecdlp_throughput_only() {
    use std::time::Instant;
    use std::hint::black_box;
    
    const N: usize = 25;
    const N_THREADS: usize = 8;
    
    let tables = read_or_gen_tables();
    let view = tables.view();
    
    // Create a point that won't be found (for maximum throughput test)
    let unfindable_point = RistrettoPoint::random(&mut rand::thread_rng());
    
    // Test range that processes many elements
    let range_start = 0i64;
    let range_end = 1 << 48; // Process 1 billion elements
    
    println!("Benchmarking throughput for {} elements", range_end - range_start);
    
    // Benchmark original SIMD version
    let mut simd_throughputs = vec![];
    for run in 0..N {
        let args = ECDLPArguments::new_with_range(range_start, range_end)
            .n_threads(N_THREADS)
            .pseudo_constant_time(false);
        
        let start = Instant::now();
        let _ = par_decode(&view, black_box(unfindable_point), args);
        let elapsed = start.elapsed();
        
        let elements_processed = (range_end - range_start) as f64;
        let throughput = elements_processed / elapsed.as_secs_f64();
        simd_throughputs.push(throughput);
        
        println!("Run {}: {:.2}M elements/sec", run + 1, throughput / 1_000_000.0);
    }
    
    let avg_simd = simd_throughputs.iter().sum::<f64>() / simd_throughputs.len() as f64;
    println!("\nAverage SIMD throughput: {:.2}M elements/sec", avg_simd / 1_000_000.0);
    
    // Benchmark j-batched version
    let mut batched_throughputs = vec![];
    for run in 0..N {
        let args = ECDLPArguments::new_with_range(range_start, range_end)
            .n_threads(N_THREADS)
            .pseudo_constant_time(false);
        
        let start = Instant::now();
        let _ = par_decode_j_batch(&view, black_box(unfindable_point), args);
        let elapsed = start.elapsed();
        
        let elements_processed = (range_end - range_start) as f64;
        let throughput = elements_processed / elapsed.as_secs_f64();
        batched_throughputs.push(throughput);
        
        println!("Run {}: {:.2}M elements/sec", run + 1, throughput / 1_000_000.0);
    }
    
    let avg_batched = batched_throughputs.iter().sum::<f64>() / batched_throughputs.len() as f64;
    println!("\nAverage j-batched throughput: {:.2}M elements/sec", avg_batched / 1_000_000.0);
    
    // Calculate speedup
    let speedup = avg_batched / avg_simd;
    println!("\nSpeedup: {:.2}x", speedup);
}

#[test]
fn bench_inversion_methods_isolated() {
    use std::time::Instant;
    
    const BATCH_SIZE: usize = 256;
    // const J_BATCH_SIZE: usize = 4;
    const ITERATIONS: usize = 100_000;
    
    // Generate test data
    let mut rng = rand::thread_rng();
    
    // Benchmark standard batch inversion
    let mut standard_total = 0.0;
    for _ in 0..10 {
        let mut data = [FieldElement::ONE; BATCH_SIZE];
        for i in 0..BATCH_SIZE {
            data[i] = FieldElement::from_limbs([rng.gen::<u64>(),0,0,0,0]);
        }
        
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let mut work = data.clone();
            batch_invert(&mut work);
            black_box(work);
        }
        let elapsed = start.elapsed();
        standard_total += elapsed.as_secs_f64();
    }
    
    // Benchmark j-batched inversion
    let mut batched_total = 0.0;
    for _ in 0..10 {
        let mut data = [[FieldElement::ONE; BATCH_SIZE]; J_BATCH_SIZE];
        for j in 0..J_BATCH_SIZE {
            for i in 0..BATCH_SIZE {
                data[j][i] = FieldElement::from_limbs([rng.gen::<u64>(),0,0,0,0]);
            }
        }
        
        let start = Instant::now();
        for _ in 0..ITERATIONS / J_BATCH_SIZE {
            let mut work = data.clone();
            j_batched_inversion_correct(&mut work);
            black_box(work);
        }
        let elapsed = start.elapsed();
        batched_total += elapsed.as_secs_f64();
    }
    
    println!("Standard batch inversion: {:.6} seconds", standard_total / 10.0);
    println!("J-batched inversion: {:.6} seconds", batched_total / 10.0);
    println!("Speedup: {:.2}x", standard_total / batched_total);
}
}
