use yarblas::*;
use rand::Rng;

#[test]
fn test_sgemm() {
    unsafe {
        let m = 1009;
        let n = 1011;
        let k = 1021;

        let mut rng = rand::thread_rng();
        let a = aligned_alloc::Alloc::new(m * k * std::mem::size_of::<f32>());
        let b = aligned_alloc::Alloc::new(n * k * std::mem::size_of::<f32>());
        let c = aligned_alloc::Alloc::new(m * n * std::mem::size_of::<f32>());
        let cref = aligned_alloc::Alloc::new(m * n * std::mem::size_of::<f32>());

        let aptr = a.ptr() as *mut f32;
        let bptr = b.ptr() as *mut f32;
        let cptr = c.ptr() as *mut f32;
        let crefptr = cref.ptr() as *mut f32;

        let context = Context::new();

        for i in 0..m * k {
            *aptr.add(i) = rng.gen();
        }

        for i in 0..n * k {
            *bptr.add(i) = rng.gen();
        }

        for i in 0..m * n {
            let crand = rng.gen();
            *cptr.add(i) = crand;
            *crefptr.add(i) = crand;
        }

        let alpha = rng.gen();
        let beta = rng.gen();

        sgemm(
            &context, false, false, m, n, k, alpha, aptr, m, bptr, k, beta, cptr, m,
        );

        yarblas_ref::sgemm(
            false, false, m, n, k, alpha, aptr, m, bptr, k, beta, crefptr, m,
        );

        for i in 0..m * n {
            let c0 = *cptr.add(i);
            let cref0 = *crefptr.add(i);
            assert!(feq(c0, cref0), "c0={};cref0={};i={}", c0, cref0, i);
        }
    }
}

fn feq(a: f32, b: f32) -> bool {
    if a == b {
        true
    } else if a == 0.0 || b == 0.0 || (a.abs() + b.abs() < std::f32::MIN_POSITIVE) {
        (a - b).abs() < std::f32::EPSILON * 10.0 * std::f32::MIN_POSITIVE
    } else {
        (a - b).abs() / (a.abs() + b.abs()) < std::f32::EPSILON * 10.0
    }
}
