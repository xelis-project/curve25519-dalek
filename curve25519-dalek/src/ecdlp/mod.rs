//! # Elliptic Curve Discrete Logarithm Problem (ECDLP)
//!
//! This file enables the decoding of integers from [`RistrettoPoint`]s. As this requires
//! a bruteforce operation, this can take quite a long time (multiple seconds) for large input spaces.
//!
//! The algorithm depends on a constant `L1` parameter, which enables changing the space/time tradeoff.
//! To use a given `L1` constant, you need to generate a precomputed tables file, or download a pre-generated one.
//! The resulting file may be quite big, which is why it is recommanded to load it at runtime using [`ecdlp::ECDLPTablesFile`].
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

use crate::{
    constants::MONTGOMERY_A_NEG, constants::RISTRETTO_BASEPOINT_POINT as G, field::FieldElement,
    RistrettoPoint, Scalar,
};
use affine_montgomery::AffineMontgomeryPoint;
use core::{
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
};

pub use table::{
    table_generation, ECDLPTablesFileView, NoOpProgressTableGenerationReportFunction,
    ProgressTableGenerationReportFunction, ReportStep,
};

use table::{BATCH_SIZE, L2};

/// A trait to represent progress report functions.
/// It is auto-implemented on any `F: Fn(f64) -> ConstrolFlow<()>`.
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

    fast_ecdlp(
        precomputed_tables,
        normalized,
        point_iter,
        args.pseudo_constant_time,
        &AtomicBool::new(false),
        args.progress_report_function,
        &t2_cache,
        &t2_cache_alpha
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
                    make_point_iterator(precomputed_tables, normalized, num_batches);

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
            .lookup(&target_montgomery.u.as_bytes(), |i| {
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
                .lookup(&qx.as_bytes(), |i| {
                    consider_candidate(j_start_shifted + i as i64) || consider_candidate(j_start_shifted - i as i64)
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

cfg_if::cfg_if! {
    if #[cfg(feature = "simd")] {
        // SIMD-enabled implementation
        fn check_batch_for_zeros(
            batch: &[FieldElement], 
            j_start: usize,
            precomputed_tables: &ECDLPTablesFileView<'_>,
            mut consider_candidate: impl FnMut(i64) -> bool,
            pseudo_constant_time: bool
        ) -> Option<bool> {
            use wide::{i8x32, CmpEq};
            use std::mem::transmute;

            let mut zero_indices = Vec::new();
            
            // SIMD zero detection
            for (i, diff) in batch.iter().enumerate() {
                let bytes = diff.as_bytes();

                // Safe because we're only checking for zero values, where 
                // the memory representation is identical between u8 and i8
                let v = unsafe { 
                    i8x32::from(*transmute::<&[u8; 32], &[i8; 32]>(&bytes))
                };
                let zero_v = i8x32::splat(0);
                
                if v.cmp_eq(zero_v).all() {
                    zero_indices.push(i + 1);
                }
            }
            
            // Process detected zeros
            for j in zero_indices {
                let found =
                    consider_candidate((j_start as i64 + j as i64) << precomputed_tables.get_l1())
                        || consider_candidate(
                            (j_start as i64 - j as i64) << precomputed_tables.get_l1(),
                        );
                if !pseudo_constant_time && found {
                    return Some(true);
                }
            }
            
            None
        }
    } else {
        // Fallback non-SIMD implementation
        fn check_batch_for_zeros(
            batch: &[FieldElement],
            j_start: usize,
            precomputed_tables: &ECDLPTablesFileView<'_>,
            mut consider_candidate: impl FnMut(i64) -> bool,
            pseudo_constant_time: bool
        ) -> Option<bool> {
            for (i, diff) in batch.iter().enumerate() {
                let j = i + 1;
                if diff.is_zero_not_ct() {
                    let found =
                        consider_candidate((j_start as i64 + j as i64) << precomputed_tables.get_l1())
                            || consider_candidate(
                                (j_start as i64 - j as i64) << precomputed_tables.get_l1(),
                            );
                    if !pseudo_constant_time && found {
                        return Some(true);
                    }
                }
            }
            
            None
        }
    }
}

fn batch_field_subtract<const N: usize>(
    batch: &mut [FieldElement; N],
    u_values: &[FieldElement; N],
    target_u: &FieldElement,
) {
    use crate::field_simd::{MAX_SIMD_WIDTH};
    
    let mut pos = 0;
    let chunk_size = if MAX_SIMD_WIDTH >= 8 { 8 }
                    else if MAX_SIMD_WIDTH >= 4 { 4 }
                    else { 2 };
    
    // Process in optimal chunks
    while pos + chunk_size <= N {
        match chunk_size {
            8 => {
                let chunk: &mut [FieldElement; 8] = (&mut batch[pos..pos+8]).try_into().unwrap();
                let u_chunk: &[FieldElement; 8] = (&u_values[pos..pos+8]).try_into().unwrap();
                *chunk = FieldElement::batch_subtract_8way(u_chunk, target_u);
            }
            4 => {
                let chunk: &mut [FieldElement; 4] = (&mut batch[pos..pos+4]).try_into().unwrap();
                let u_chunk: &[FieldElement; 4] = (&u_values[pos..pos+4]).try_into().unwrap();
                *chunk = FieldElement::batch_subtract_4way(u_chunk, target_u);
            }
            _ => unreachable!(),
        }
        pos += chunk_size;
    }
    
    // Handle remaining elements
    while pos < N {
        batch[pos] = &u_values[pos] - target_u;
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

/// Non-SIMD fallback implementation
#[cfg(not(feature = "simd"))]
pub fn batch_i64_to_scalar(inputs: &[i64], outputs: &mut [Scalar]) {
    for (i, &value) in inputs.iter().enumerate() {
        outputs[i] = i64_to_scalar(value);
    }
}

pub fn batch_lookup<const N: usize>(
    t1_table: &table::CuckooT1HashMapView<'_>,  // Correct type from error message
    qxs: &[FieldElement; N],
    base_shifts: &[i64; N],
    candidates_buffer: &mut Vec<i64>
) {
    // Process all qxs in parallel, storing matches
    for i in 0..N {
        let j_start_shifted = base_shifts[i];
        
        // Use the correct lookup method
        t1_table.lookup(&qxs[i].as_bytes(), |idx| {
            // Add both candidate forms to the buffer
            candidates_buffer.push(j_start_shifted + idx as i64);
            candidates_buffer.push(j_start_shifted - idx as i64);
            
            // Return false to continue collecting all matches
            false
        });
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

        // Case 0: target is 0. Has to be handled separately.
        let j_start_shifted = (j_start as i64) << l1;
        if target_montgomery.is_identity_not_ct() {
            consider_candidate(j_start_shifted);
            if !pseudo_constant_time {
                break 'outer;
            }
        }

        // Case 2: j=0. Has to be handled separately.
        if t1_table
            .lookup(&target_montgomery.u.as_bytes(), |i| {
                consider_candidate(j_start_shifted + i as i64)
                    || consider_candidate(j_start_shifted - i as i64)
            })
            .is_some()
            && !pseudo_constant_time
        {
            break 'outer;
        }

        // Use SIMD batch subtraction instead of the for loop
        batch_field_subtract(batch, &t2_u_values, &target_montgomery.u);

        // Check for exceptional cases (when diff is zero)
        if let Some(true) = check_batch_for_zeros(
            batch, 
            j_start, 
            precomputed_tables,
            |candidate| consider_candidate(candidate),
            pseudo_constant_time
        ) {
            break 'outer;
        }

        // nu = Z^-1
        FieldElement::batch_invert(batch);
    
        batch_field_subtract(
                  alphas,
                  &t2_cache_alpha,
                  &target_montgomery.u
        );

        // Batch compute qxs for the regular case
        // lambda = (T2[j]_y - Pm_y) * nu
        FieldElement::batch_compute_qx(qxs, 
                  &t2_vs, 
                  &target_montgomery.v, 
                  batch,
                  alphas);

        for (idx, qx) in qxs.iter().enumerate() {
            let j = idx + 1;
            let j_start_shifted = (j_start as i64 - j as i64) << l1;
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
        }

        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }

        FieldElement::batch_compute_qx(neg_qxs,
            &t2_vs_neg,
            &target_montgomery.v,
            batch,
            alphas);

        for (idx, qx) in neg_qxs.iter().enumerate() {
            let j = idx + 1;
            let j_start_shifted = (j_start as i64 + j as i64) << l1;
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
        }
    }

    found
}

pub fn approximate_decode_bit_length(point: &RistrettoPoint) -> usize {
  let mut guess_bits = 1;
  let mut base = Scalar::from(1u64);
  let mut guess = RistrettoPoint::default();

  while guess != *point {
      base = base + base; // double the scalar each iteration
      guess = RistrettoPoint::mul_base(&base);
      guess_bits += 1;

      // Fail-safe cap to prevent infinite loops
      if guess_bits > 64 {
          break;
      }
  }

  guess_bits
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

    const L1: usize = 29;

    // Necessary for one ECDLP tables allocation only
    static TABLES: Mutex<Option<Arc<ECDLPTables>>> = Mutex::new(None);

    fn read_or_gen_tables() -> Arc<ECDLPTables> {
        let mut tables = TABLES.lock().expect("acquire tables lock");
        if let Some(v) = tables.as_ref().cloned() {
            return v;
        }

        let inner = if !Path::new("ecdlp_table.bin").exists() {
            let tables = ECDLPTables::generate(L1).unwrap();
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
    
        let tables = read_or_gen_tables();
        let view = tables.view();
    
        let value: u64 = 1_000_000_000 * 100_000_000;
        // let value: u64 = 9_223_372_036_854_775_806;
        let mut point = RistrettoPoint::mul_base(&Scalar::from(value));
    
        if rand::thread_rng().gen() {
            // Optionally alter the point via compression/decompression
            point = point.compress().decompress().unwrap();
        }
    
        let args = ECDLPArguments::new_with_range(0, 1 << 63)
            .n_threads(16)
            .pseudo_constant_time(false);
    
        let now = Instant::now();
    
        let res = par_decode(&view, point, args);
    
        let elapsed = now.elapsed();
        println!(
            "Decoded value: {:?} in {:?} seconds",
            res,
            elapsed.as_secs_f64()
        );
    
        assert_eq!(res, Some(value as i64));
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
}
