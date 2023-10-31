use nalgebra::DMatrix;
use num::complex::Complex64;

// pub type SMatrix<T, const N: usize> = SMatrix<T, Const<N>, Const<N>>;

// pub type FMatrix<const N: usize> = SMatrix<f64, N>;
// pub type CMatrix<const N: usize> = SMatrix<Complex64, N>;


pub type FMatrix = DMatrix<f64>;
pub type CMatrix = DMatrix<Complex64>;