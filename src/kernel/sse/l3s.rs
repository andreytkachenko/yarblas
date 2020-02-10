use super::fma::fmadd_ps;
use super::intrinsics::*;
use crate::kernel::params::single::NR;
use crate::matrix::{Matrix, MatrixMut, MutMatrix};

pub(crate) unsafe fn sgemm_sup_1x5<A: Matrix<f32>, C: MatrixMut<f32>>(
    k: usize,
    alpha: f32,
    a: A,
    pb: MutMatrix<f32>,
    beta: f32,
    c: C,
) {
    let mut c0_3 = _mm_setzero_ps();
    let mut c4 = 0.0f32;

    let mut a = a;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = *a.ptr();
        let a0_simd = _mm_broadcast_ss(&*a.ptr());
        
        c0_3 = fmadd_ps(_mm_loadu_ps(pb.ptr()), a0_simd, c0_3);
        c4 += *pb.col(4) * a0;

        a.inc_row();
        pb.shift_col(NR);
    }

    // c0 *= alpha;
    // c1 *= alpha;
    // c2 *= alpha;
    // c3 *= alpha;
    // c4 *= alpha;
    
    let ccol0 = c.ptr_mut();
    let ccol1 = c.row_mut(1);
    let ccol2 = c.row_mut(2);
    let ccol3 = c.row_mut(3);
    let ccol4 = c.row_mut(4);

    // if beta != 0.0 {
    //     c0 += beta * *ccol0;
    //     c1 += beta * *ccol1;
    //     c2 += beta * *ccol2;
    //     c3 += beta * *ccol3;
    //     c4 += beta * *ccol4;
    // }

    *ccol0 += std::mem::transmute::<_, f32>(_mm_extract_ps(c0_3, 0));
    *ccol1 += std::mem::transmute::<_, f32>(_mm_extract_ps(c0_3, 1));
    *ccol2 += std::mem::transmute::<_, f32>(_mm_extract_ps(c0_3, 2));
    *ccol3 += std::mem::transmute::<_, f32>(_mm_extract_ps(c0_3, 3));
    *ccol4 += c4;
}

pub(crate) unsafe fn sgemm_pb_x8(k: usize, b: *const f32, ldb: usize, pb: *mut f32) {
    let mut bcol0 = b;
    let mut bcol1 = b.add(ldb);
    let mut bcol2 = b.add(ldb * 2);
    let mut bcol3 = b.add(ldb * 3);
    let mut bcol4 = b.add(ldb * 4);

    let mut pb = pb;

    for _ in 0..k {
        _mm_storeu_ps(pb, _mm_set_ps(*bcol3, *bcol2, *bcol1, *bcol0));
        *pb.add(4) = *bcol4;

        bcol0 = bcol0.add(1);
        bcol1 = bcol1.add(1);
        bcol2 = bcol2.add(1);
        bcol3 = bcol3.add(1);
        bcol4 = bcol4.add(1);

        pb = pb.add(NR);
    }
}