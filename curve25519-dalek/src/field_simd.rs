use cfg_if::cfg_if;
#[cfg(feature = "simd")]
cfg_if! {
  if #[cfg(curve25519_dalek_bits = "32")] {
      #[allow(unused_imports)]
      use wide::{u32x8, u32x4, u32x2};
  } else {
      #[allow(unused_imports)]
      use wide::{u64x4};
  }
}

cfg_if! {
    if #[cfg(all(curve25519_dalek_backend = "fiat", curve25519_dalek_bits = "32"))] {
        pub const MAX_SIMD_WIDTH: usize = 8;
    } else if #[cfg(all(curve25519_dalek_backend = "fiat", curve25519_dalek_bits = "64"))] {
        pub const MAX_SIMD_WIDTH: usize = 4;
    } else if #[cfg(curve25519_dalek_bits = "64")] {
        pub const MAX_SIMD_WIDTH: usize = 4;
    } else {
        pub const MAX_SIMD_WIDTH: usize = 8;
    }
}

use crate::field::FieldElement;

// Now implement for the specific type
impl FieldElement {
    // 4-way SIMD operations
    #[inline]
    pub(crate) fn batch_subtract_4way(_batch: &[Self; 4], _target: &Self) -> [Self; 4] {
        cfg_if! {
            if #[cfg(curve25519_dalek_bits = "64")] {
                // 64-bit implementation
                let mut results = [Self::ZERO; 4];
                
                const OFFSET_0: u64 = 36028797018963664u64;
                const OFFSET_1_4: u64 = 36028797018963952u64;
                
                for limb_idx in 0..5 {
                    let batch_limbs = u64x4::from([
                        _batch[0].0[limb_idx],
                        _batch[1].0[limb_idx],
                        _batch[2].0[limb_idx],
                        _batch[3].0[limb_idx],
                    ]);
                    
                    let target_limb = u64x4::splat(_target.0[limb_idx]);
                    let offset = if limb_idx == 0 {
                        u64x4::splat(OFFSET_0)
                    } else {
                        u64x4::splat(OFFSET_1_4)
                    };
                    
                    let diff = batch_limbs + offset - target_limb;
                    let diff_array = diff.to_array();
                    
                    for i in 0..4 {
                        results[i].0[limb_idx] = diff_array[i];
                    }
                }
                
                // Use Sub trait instead of private reduce method  
                [
                    &results[0] - &Self::ZERO,  // This triggers field reduction
                    &results[1] - &Self::ZERO,
                    &results[2] - &Self::ZERO,
                    &results[3] - &Self::ZERO,
                ]
            } else {
                // 32-bit implementation
                let mut results = [Self::ZERO; 4];
                
                for limb_idx in 0..10 {
                    let batch_limbs = u32x4::from([
                        batch[0].0[limb_idx],
                        batch[1].0[limb_idx],
                        batch[2].0[limb_idx],
                        batch[3].0[limb_idx],
                    ]);
                    
                    let target_limb = u32x4::splat(target.0[limb_idx]);
                    
                    let offset = if limb_idx % 2 == 0 {
                        u32x4::splat(0x3ffffff << 4)
                    } else {
                        u32x4::splat(0x1ffffff << 4)
                    };
                    
                    let offset = if limb_idx == 0 {
                        u32x4::splat(0x3ffffed << 4)
                    } else {
                        offset
                    };
                    
                    let diff = batch_limbs + offset - target_limb;
                    let diff_array = diff.to_array();
                    
                    for i in 0..4 {
                        results[i].0[limb_idx] = diff_array[i];
                    }
                }
                
                // Use Sub trait instead of private reduce method
                [
                    &results[0] - &Self::ZERO,
                    &results[1] - &Self::ZERO,
                    &results[2] - &Self::ZERO,
                    &results[3] - &Self::ZERO,
                ]
            }
        }
    }
    #[inline]
    pub(crate) fn batch_add_4way(_batch: &[Self; 4], _target: &Self) -> [Self; 4] {
        cfg_if! {
            if #[cfg(curve25519_dalek_bits = "64")] {
                // 64-bit implementation
                let mut results = [Self::ZERO; 4];
                
                for limb_idx in 0..5 {
                    let batch_limbs = u64x4::from([
                        _batch[0].0[limb_idx],
                        _batch[1].0[limb_idx],
                        _batch[2].0[limb_idx],
                        _batch[3].0[limb_idx],
                    ]);
                    
                    let target_limb = u64x4::splat(_target.0[limb_idx]);
                    
                    // Simple addition of the limbs
                    let sum = batch_limbs + target_limb;
                    let sum_array = sum.to_array();
                    
                    for i in 0..4 {
                        results[i].0[limb_idx] = sum_array[i];
                    }
                }
                
                // Use Add trait to trigger field reduction  
                [
                    &results[0] + &Self::ZERO,
                    &results[1] + &Self::ZERO,
                    &results[2] + &Self::ZERO,
                    &results[3] + &Self::ZERO,
                ]
            } else {
                // 32-bit implementation
                let mut results = [Self::ZERO; 4];
                
                for limb_idx in 0..10 {
                    let batch_limbs = u32x4::from([
                        _batch[0].0[limb_idx],
                        _batch[1].0[limb_idx],
                        _batch[2].0[limb_idx],
                        _batch[3].0[limb_idx],
                    ]);
                    
                    let target_limb = u32x4::splat(_target.0[limb_idx]);
                    
                    // Simple addition of the limbs
                    let sum = batch_limbs + target_limb;
                    let sum_array = sum.to_array();
                    
                    for i in 0..4 {
                        results[i].0[limb_idx] = sum_array[i];
                    }
                }
                
                // Use Add trait to trigger field reduction
                [
                    &results[0] + &Self::ZERO,
                    &results[1] + &Self::ZERO,
                    &results[2] + &Self::ZERO,
                    &results[3] + &Self::ZERO,
                ]
            }
        }
    }
    #[inline]
    pub(crate) fn batch_vecadd_4way(_a: &[Self; 4], _b: &[Self; 4]) -> [Self; 4] {
        cfg_if! {
            if #[cfg(curve25519_dalek_bits = "64")] {
                // 64-bit implementation
                let mut results = [Self::ZERO; 4];
                
                for limb_idx in 0..5 {
                    let a_limbs = u64x4::from([
                        _a[0].0[limb_idx],
                        _a[1].0[limb_idx],
                        _a[2].0[limb_idx],
                        _a[3].0[limb_idx],
                    ]);
                    
                    let b_limbs = u64x4::from([
                        _b[0].0[limb_idx],
                        _b[1].0[limb_idx],
                        _b[2].0[limb_idx],
                        _b[3].0[limb_idx],
                    ]);
                    
                    // Simple addition of the limbs
                    let sum = a_limbs + b_limbs;
                    let sum_array = sum.to_array();
                    
                    for i in 0..4 {
                        results[i].0[limb_idx] = sum_array[i];
                    }
                }
                
                // Use Add trait to trigger field reduction  
                [
                    &results[0] + &Self::ZERO,
                    &results[1] + &Self::ZERO,
                    &results[2] + &Self::ZERO,
                    &results[3] + &Self::ZERO,
                ]
            } else {
                // 32-bit implementation
                let mut results = [Self::ZERO; 4];
                
                for limb_idx in 0..10 {
                    let a_limbs = u32x4::from([
                        _a[0].0[limb_idx],
                        _a[1].0[limb_idx],
                        _a[2].0[limb_idx],
                        _a[3].0[limb_idx],
                    ]);
                    
                    let b_limbs = u32x4::from([
                        _b[0].0[limb_idx],
                        _b[1].0[limb_idx],
                        _b[2].0[limb_idx],
                        _b[3].0[limb_idx],
                    ]);
                    
                    // Simple addition of the limbs
                    let sum = a_limbs + b_limbs;
                    let sum_array = sum.to_array();
                    
                    for i in 0..4 {
                        results[i].0[limb_idx] = sum_array[i];
                    }
                }
                
                // Use Add trait to trigger field reduction
                [
                    &results[0] + &Self::ZERO,
                    &results[1] + &Self::ZERO,
                    &results[2] + &Self::ZERO,
                    &results[3] + &Self::ZERO,
                ]
            }
        }
    }
    
    // 8-way SIMD operations (only for 32-bit fields)
    #[inline]
    pub(crate) fn batch_subtract_8way(_batch: &[Self; 8], _target: &Self) -> [Self; 8] {
        cfg_if! {
            if #[cfg(curve25519_dalek_bits = "32")] {
                // 32-bit can handle 8-way operations
                let mut results = [Self::ZERO; 8];
                
                for limb_idx in 0..10 {
                    let batch_limbs = u32x8::from([
                        _batch[0].0[limb_idx],
                        _batch[1].0[limb_idx],
                        _batch[2].0[limb_idx],
                        _batch[3].0[limb_idx],
                        _batch[4].0[limb_idx],
                        _batch[5].0[limb_idx],
                        _batch[6].0[limb_idx],
                        _batch[7].0[limb_idx],
                    ]);
                    
                    let target_limb = u32x8::splat(_target.0[limb_idx]);
                    
                    let offset = if limb_idx % 2 == 0 {
                        u32x8::splat(0x3ffffff << 4)
                    } else {
                        u32x8::splat(0x1ffffff << 4)
                    };
                    
                    let offset = if limb_idx == 0 {
                        u32x8::splat(0x3ffffed << 4)
                    } else {
                        offset
                    };
                    
                    let diff = batch_limbs + offset - target_limb;
                    let diff_array = diff.to_array();
                    
                    for i in 0..8 {
                        results[i].0[limb_idx] = diff_array[i];
                    }
                }
                
                // Use Sub trait instead of private reduce method
                results.map(|r| &r - &Self::ZERO)
            } else {
                // 64-bit doesn't support 8-way, return error
                unimplemented!("8-way not supported for 64-bit field elements")
            }
        }
    }
    #[inline]
    pub(crate) fn batch_add_8way(_batch: &[Self; 8], _target: &Self) -> [Self; 8] {
        cfg_if! {
            if #[cfg(curve25519_dalek_bits = "32")] {
                // 32-bit can handle 8-way operations
                let mut results = [Self::ZERO; 8];
                
                for limb_idx in 0..10 {
                    let batch_limbs = u32x8::from([
                        _batch[0].0[limb_idx],
                        _batch[1].0[limb_idx],
                        _batch[2].0[limb_idx],
                        _batch[3].0[limb_idx],
                        _batch[4].0[limb_idx],
                        _batch[5].0[limb_idx],
                        _batch[6].0[limb_idx],
                        _batch[7].0[limb_idx],
                    ]);
                    
                    let target_limb = u32x8::splat(_target.0[limb_idx]);
                    
                    // Simple addition of the limbs
                    let sum = batch_limbs + target_limb;
                    let sum_array = sum.to_array();
                    
                    for i in 0..8 {
                        results[i].0[limb_idx] = sum_array[i];
                    }
                }
                
                // Use Add trait to trigger field reduction
                results.map(|r| &r + &Self::ZERO)
            } else {
                // 64-bit doesn't support 8-way, return error
                unimplemented!("8-way not supported for 64-bit field elements")
            }
        }
    }
    #[inline]
    pub(crate) fn batch_vecadd_8way(_a: &[Self; 8], _b: &[Self; 8]) -> [Self; 8] {
        cfg_if! {
            if #[cfg(curve25519_dalek_bits = "32")] {
                // 32-bit can handle 8-way operations
                let mut results = [Self::ZERO; 8];
                
                for limb_idx in 0..10 {
                    let a_limbs = u32x8::from([
                        _a[0].0[limb_idx],
                        _a[1].0[limb_idx],
                        _a[2].0[limb_idx],
                        _a[3].0[limb_idx],
                        _a[4].0[limb_idx],
                        _a[5].0[limb_idx],
                        _a[6].0[limb_idx],
                        _a[7].0[limb_idx],
                    ]);
                    
                    let b_limbs = u32x8::from([
                        _b[0].0[limb_idx],
                        _b[1].0[limb_idx],
                        _b[2].0[limb_idx],
                        _b[3].0[limb_idx],
                        _b[4].0[limb_idx],
                        _b[5].0[limb_idx],
                        _b[6].0[limb_idx],
                        _b[7].0[limb_idx],
                    ]);
                    
                    // Simple addition of the limbs
                    let sum = a_limbs + b_limbs;
                    let sum_array = sum.to_array();
                    
                    for i in 0..8 {
                        results[i].0[limb_idx] = sum_array[i];
                    }
                }
                
                // Use Add trait to trigger field reduction
                results.map(|r| &r + &Self::ZERO)
            } else {
                // 64-bit doesn't support 8-way, return error
                unimplemented!("8-way not supported for 64-bit field elements")
            }
        }
    }
    
    /// Efficiently computes qx values for a batch of field elements using SIMD operations.
    /// 
    /// This function calculates qx = (v_diff * nu)² + alpha for all elements in the batch.
    /// The calculation is optimized using SIMD operations where available.
    /// 
    /// # Arguments
    /// * `qxs` - Output array to store computed qx values
    /// * `t2_point_vs` - Array of T2[j].v values (y-coordinates)
    /// * `target_v` - The target point's v value (Pm.v)
    /// * `nus` - Array of nu values (previously computed as inversions)
    /// * `alphas` - Array of alpha values
    /// 
    /// # Performance
    /// The function processes elements in batches of 8 or 4 depending on SIMD support,
    /// with a fallback to scalar operations for any remaining elements.
    #[inline]
    pub(crate) fn batch_compute_qx<const N: usize>(
        qxs: &mut [FieldElement; N],  // Output array
        t2_point_vs: &[FieldElement; N],  // T2[j]_y values
        target_v: &FieldElement,       // Target value
        nus: &[FieldElement; N],       // nu values (previously calculated inversions)
        alphas: &[FieldElement; N],    // alpha values
    ) {
        let len = N;  // Use the const generic parameter directly
        
        let mut pos = 0;
        
        // Process 8-element chunks if supported
        while pos + 8 <= len && MAX_SIMD_WIDTH >= 8 {
            // Create fixed-size array slices for the batch operation
            let v_chunk: &[FieldElement; 8] = (&t2_point_vs[pos..pos+8]).try_into().unwrap();
            let nu_chunk: &[FieldElement; 8] = (&nus[pos..pos+8]).try_into().unwrap();
            let alpha_chunk: &[FieldElement; 8] = (&alphas[pos..pos+8]).try_into().unwrap();
            let qx_chunk: &mut [FieldElement; 8] = (&mut qxs[pos..pos+8]).try_into().unwrap();
            
            // 1. Compute differences using batch subtract: v_diff = t2_point_v - target_v
            let diff = FieldElement::batch_subtract_8way(v_chunk, target_v);
            
            // 2. Multiply each difference by its nu (temporarily using scalar for now)
            for i in 0..8 {
                qx_chunk[i] = &diff[i] * &nu_chunk[i];
                qx_chunk[i] = qx_chunk[i].square();
            }
    
            // 3. Add alpha values using SIMD batch addition
            *qx_chunk = FieldElement::batch_vecadd_8way(qx_chunk, alpha_chunk);
            
            pos += 8;
        }
        
        // Process 4-element chunks
        while pos + 4 <= len && MAX_SIMD_WIDTH >= 4 {
            let v_chunk: &[FieldElement; 4] = (&t2_point_vs[pos..pos+4]).try_into().unwrap();
            let nu_chunk: &[FieldElement; 4] = (&nus[pos..pos+4]).try_into().unwrap();
            let alpha_chunk: &[FieldElement; 4] = (&alphas[pos..pos+4]).try_into().unwrap();
            let qx_chunk: &mut [FieldElement; 4] = (&mut qxs[pos..pos+4]).try_into().unwrap();
            
            // Same steps as above but with 4-way SIMD
            let diff = FieldElement::batch_subtract_4way(v_chunk, target_v);
            
            for i in 0..4 {
                qx_chunk[i] = &diff[i] * &nu_chunk[i];
                qx_chunk[i] = qx_chunk[i].square();
            }
    
            *qx_chunk = FieldElement::batch_vecadd_4way(qx_chunk, alpha_chunk);
            
            pos += 4;
        }
        
        // Process remaining elements with scalar operations
        while pos < len {
            // v_diff = t2_point_v - target_v
            qxs[pos] = &t2_point_vs[pos] - target_v;
            
            // lambda = v_diff * nu
            qxs[pos] = &qxs[pos] * &nus[pos];
            
            // qx = lambda² + alpha
            qxs[pos] = qxs[pos].square();
            qxs[pos] = &qxs[pos] + &alphas[pos];
            
            pos += 1;
        }
    }
}