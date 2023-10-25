use nalgebra::Const;
use num::complex::Complex64;

pub type SMatrix<T, const N: usize> = nalgebra::OMatrix<T, Const<N>, Const<N>>;

pub type FMatrix<const N: usize> = SMatrix<f64, N>;
pub type CMatrix<const N: usize> = SMatrix<Complex64, N>;