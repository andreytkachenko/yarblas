use std::marker::PhantomData;
use crate::kernel::params::single::{MR, NR};
pub struct Context<F> {
    pub num_threads: usize,
    pub mc: usize,
    pub nc: usize,
    pub kc: usize,
    pub avx: bool,
    pub avx512: bool,
    pub sse: bool,
    pub fma: bool,
    pub transposed: bool,
    _m: PhantomData<F>,
}

impl<F> Context<F> {
    pub fn new(m: usize, n: usize, k: usize) -> Context<F> {
        let (threads_count, mc, kc, nc, avx, avx512, sse, fma) = cpuid::identify().ok()
            .map(|info| {
                let nc = 0;
                let mc = 0;
                let kc = 0;
                
                (
                    info.total_logical_cpus as usize,
                    mc, kc, nc, 
                    false, false, false, false//info.has_feature()
                )
            })
            .unwrap_or((1, 12, 16, 1024, true, false, true, true));

        Context {
            num_threads: threads_count,
            transposed: false,
            mc: mc * MR,
            kc: kc * MR,
            nc: nc * NR,
            avx: true, 
            avx512: false,
            sse: true,
            fma: true,
            _m: Default::default(),
        }
    }
}