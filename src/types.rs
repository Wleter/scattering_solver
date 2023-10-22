use num::complex::Complex64;

pub type SMatrix<T, const N: usize> = nalgebra::SMatrix<T, N, N>;

pub type FMatrix<const N: usize> = SMatrix<f64, N>;
pub type CMatrix<const N: usize> = SMatrix<Complex64, N>;
