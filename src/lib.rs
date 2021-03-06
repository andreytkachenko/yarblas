pub mod matrix;
pub mod gemm;
mod sgemm;
pub mod aligned_alloc;
pub mod kernel;
pub mod dim;
pub mod executor;
pub mod context;

#[cfg(test)]
extern crate blas;
#[cfg(test)]
extern crate openblas;
#[cfg(test)]
mod test;


pub use crate::sgemm::sgemm;
