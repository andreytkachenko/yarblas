use super::fma::fmadd_ps;
use super::intrinsics::*;
use crate::kernel::params::single::{MR, NR};
use crate::matrix::{Matrix, MutMatrix, MatrixMut};
use crunchy::unroll;

#[inline]
pub(crate) unsafe fn sgemm_ukr_16x5<C: MatrixMut<f32>>(
    k: usize,
    alpha: f32,
    pa: MutMatrix<f32>,
    pb: MutMatrix<f32>,
    beta: f32,
    c: C,
) {
    let mut mt00 = _mm256_setzero_ps();
    let mut mt01 = _mm256_setzero_ps();
    let mut mt02 = _mm256_setzero_ps();
    let mut mt03 = _mm256_setzero_ps();
    let mut mt04 = _mm256_setzero_ps();

    let mut mt10 = _mm256_setzero_ps();
    let mut mt11 = _mm256_setzero_ps();
    let mut mt12 = _mm256_setzero_ps();
    let mut mt13 = _mm256_setzero_ps();
    let mut mt14 = _mm256_setzero_ps();
    
    let mut pa = pa.ptr();
    let mut pb = pb.ptr();

    const BATCH: usize = 8;

    let k_right = k % BATCH;
    let k_main = k - k_right;

    for _ in (0..k_main).step_by(BATCH) {
        unroll! {
            for i in 0..8 {
                let a0 = _mm256_load_ps(pa.add(i * MR));
                let a1 = _mm256_load_ps(pa.add(i * MR + 8));

                let b0 = _mm256_broadcast_ss(&*pb.add(i * NR));
                let b1 = _mm256_broadcast_ss(&*pb.add(i * NR + 1));
                let b2 = _mm256_broadcast_ss(&*pb.add(i * NR + 2));
                let b3 = _mm256_broadcast_ss(&*pb.add(i * NR + 3));
                let b4 = _mm256_broadcast_ss(&*pb.add(i * NR + 4));
                
                mt00 = fmadd_ps(a0, b0, mt00);
                mt01 = fmadd_ps(a0, b1, mt01);
                mt02 = fmadd_ps(a0, b2, mt02);
                mt03 = fmadd_ps(a0, b3, mt03);
                mt04 = fmadd_ps(a0, b4, mt04);

                mt10 = fmadd_ps(a1, b0, mt10);
                mt11 = fmadd_ps(a1, b1, mt11);
                mt12 = fmadd_ps(a1, b2, mt12);
                mt13 = fmadd_ps(a1, b3, mt13);
                mt14 = fmadd_ps(a1, b4, mt14);
            }
        }

        pa = pa.add(BATCH * MR);
        pb = pb.add(BATCH * NR);
    }

    for _ in k_main..k {
        let a0 = _mm256_load_ps(pa);
        let a1 = _mm256_load_ps(pa.add(8));

        let b0 = _mm256_broadcast_ss(&*pb);
        let b1 = _mm256_broadcast_ss(&*pb.add(1));
        let b2 = _mm256_broadcast_ss(&*pb.add(2));
        let b3 = _mm256_broadcast_ss(&*pb.add(3));
        let b4 = _mm256_broadcast_ss(&*pb.add(4));
        
        mt00 = fmadd_ps(a0, b0, mt00);
        mt01 = fmadd_ps(a0, b1, mt01);
        mt02 = fmadd_ps(a0, b2, mt02);
        mt03 = fmadd_ps(a0, b3, mt03);
        mt04 = fmadd_ps(a0, b4, mt04);

        mt10 = fmadd_ps(a1, b0, mt10);
        mt11 = fmadd_ps(a1, b1, mt11);
        mt12 = fmadd_ps(a1, b2, mt12);
        mt13 = fmadd_ps(a1, b3, mt13);
        mt14 = fmadd_ps(a1, b4, mt14);

        pa = pa.add(MR);
        pb = pb.add(NR);
    }

    // let alpha = _mm256_broadcast_ss(&alpha);

    // mt00 = _mm256_mul_ps(alpha, mt00);
    // mt01 = _mm256_mul_ps(alpha, mt01);
    // mt02 = _mm256_mul_ps(alpha, mt02);
    // mt03 = _mm256_mul_ps(alpha, mt03);
    // mt04 = _mm256_mul_ps(alpha, mt04);

    // mt10 = _mm256_mul_ps(alpha, mt10);
    // mt11 = _mm256_mul_ps(alpha, mt11);
    // mt12 = _mm256_mul_ps(alpha, mt12);
    // mt13 = _mm256_mul_ps(alpha, mt13);
    // mt14 = _mm256_mul_ps(alpha, mt14);

    let ccol0 = c.ptr_mut();
    let ccol1 = c.row_mut(1);
    let ccol2 = c.row_mut(2);
    let ccol3 = c.row_mut(3);
    let ccol4 = c.row_mut(4);

    mt00 = _mm256_add_ps(_mm256_loadu_ps(ccol0), mt00);
    mt01 = _mm256_add_ps(_mm256_loadu_ps(ccol1), mt01);
    mt02 = _mm256_add_ps(_mm256_loadu_ps(ccol2), mt02);
    mt03 = _mm256_add_ps(_mm256_loadu_ps(ccol3), mt03);
    mt04 = _mm256_add_ps(_mm256_loadu_ps(ccol4), mt04);

    mt10 = _mm256_add_ps(_mm256_loadu_ps(ccol0.add(8)), mt10);
    mt11 = _mm256_add_ps(_mm256_loadu_ps(ccol1.add(8)), mt11);
    mt12 = _mm256_add_ps(_mm256_loadu_ps(ccol2.add(8)), mt12);
    mt13 = _mm256_add_ps(_mm256_loadu_ps(ccol3.add(8)), mt13);
    mt14 = _mm256_add_ps(_mm256_loadu_ps(ccol4.add(8)), mt14);

    // if beta != 0.0 {
    //     let beta = _mm256_broadcast_ss(&beta);

    //     mt00 = fmadd_ps(beta, _mm256_loadu_ps(ccol0), mt00);
    //     mt01 = fmadd_ps(beta, _mm256_loadu_ps(ccol1), mt01);
    //     mt02 = fmadd_ps(beta, _mm256_loadu_ps(ccol2), mt02);
    //     mt03 = fmadd_ps(beta, _mm256_loadu_ps(ccol3), mt03);
    //     mt04 = fmadd_ps(beta, _mm256_loadu_ps(ccol4), mt04);
        
    //     mt10 = fmadd_ps(beta, _mm256_loadu_ps(ccol0.add(8)), mt10);
    //     mt11 = fmadd_ps(beta, _mm256_loadu_ps(ccol1.add(8)), mt11);
    //     mt12 = fmadd_ps(beta, _mm256_loadu_ps(ccol2.add(8)), mt12);
    //     mt13 = fmadd_ps(beta, _mm256_loadu_ps(ccol3.add(8)), mt13);
    //     mt14 = fmadd_ps(beta, _mm256_loadu_ps(ccol4.add(8)), mt14);
    // }

    _mm256_storeu_ps(ccol0, mt00);
    _mm256_storeu_ps(ccol1, mt01);
    _mm256_storeu_ps(ccol2, mt02);
    _mm256_storeu_ps(ccol3, mt03);
    _mm256_storeu_ps(ccol4, mt04);
    
    _mm256_storeu_ps(ccol0.add(8), mt10);
    _mm256_storeu_ps(ccol1.add(8), mt11);
    _mm256_storeu_ps(ccol2.add(8), mt12);
    _mm256_storeu_ps(ccol3.add(8), mt13);
    _mm256_storeu_ps(ccol4.add(8), mt14);
}

pub(crate) unsafe fn sgemm_sup_16x1<B: Matrix<f32>, C: MatrixMut<f32>>(
    k: usize,
    alpha: f32,
    pa: MutMatrix<f32>,
    b: B,
    beta: f32,
    c: C,
) {
    let mut mt0 = _mm256_setzero_ps();
    let mut mt1 = _mm256_setzero_ps();

    let mut pa = pa;
    let mut b = b;

    const BATCH: usize = 8;

    let k_right = k % BATCH;
    let k_main = k - k_right;

    for _ in (0..k_main).step_by(BATCH) {
        unroll! {
            for i in 0..8 {
                let a0 = _mm256_load_ps(pa.ptr());
                let a1 = _mm256_load_ps(pa.col(8));

                let b0 = _mm256_broadcast_ss(&*b.ptr());

                mt0 = fmadd_ps(a0, b0, mt0);
                mt1 = fmadd_ps(a1, b0, mt1);

                pa.shift_col(16);
                b.inc_col();
            }
        }
    }

    for _ in k_main..k {
        let a0 = _mm256_load_ps(pa.ptr());
        let a1 = _mm256_load_ps(pa.col(8));

        let b0 = _mm256_broadcast_ss(&*b.ptr());

        mt0 = fmadd_ps(a0, b0, mt0);
        mt1 = fmadd_ps(a1, b0, mt1);

        pa.shift_col(16);
        b.inc_col();
    }

    // let alpha = _mm256_broadcast_ss(&alpha);

    // mt0 = _mm256_mul_ps(alpha, mt0);
    // mt1 = _mm256_mul_ps(alpha, mt1);

    let ccol0 = c.ptr_mut();
    let ccol1 = c.ptr_mut().add(8);

    // if beta != 0.0 {
    //     let beta = _mm256_broadcast_ss(&beta);

    //     mt0 = fmadd_ps(beta, _mm256_loadu_ps(c), mt0);
    //     mt1 = fmadd_ps(beta, _mm256_loadu_ps(c.add(8)), mt1);
    // }

    mt0 = _mm256_add_ps(_mm256_loadu_ps(ccol0), mt0);
    mt1 = _mm256_add_ps(_mm256_loadu_ps(ccol1), mt1);

    _mm256_storeu_ps(ccol0, mt0);
    _mm256_storeu_ps(ccol1, mt1);
}

pub(crate) unsafe fn sgemm_pa_16x(k: usize, a: *const f32, lda: usize, pa: *mut f32) {
    let mut a = a;
    let mut pa = pa;

    for _ in 0..k {
        _mm256_store_ps(pa, _mm256_loadu_ps(a));
        _mm256_store_ps(pa.add(8), _mm256_loadu_ps(a.add(8)));

        pa = pa.add(16);
        a = a.add(lda);
    }
}