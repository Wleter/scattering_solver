use std::{iter::Sum, ops::Index};

use nalgebra::Const;
use num::complex::Complex64;
use num_traits::{One, Zero};

pub type SMatrix<T, const N: usize> = nalgebra::OMatrix<T, Const<N>, Const<N>>;

pub type FMatrix<const N: usize> = SMatrix<f64, N>;
pub type CMatrix<const N: usize> = SMatrix<Complex64, N>;

pub trait MultiField: Clone + Send + Sync + Sum + One + Zero {}
impl<const N: usize> MultiField for CMatrix<N> {}
impl<const N: usize> MultiField for FMatrix<N> {}

pub trait CMultiField: MultiField + Index<(usize, usize), Output = Complex64> {}
impl<const N: usize> CMultiField for CMatrix<N> {}
pub trait FMultiField: MultiField + Index<(usize, usize), Output = f64> {}
impl<const N: usize> FMultiField for FMatrix<N> {}


pub trait CNField<const N: usize>: CMultiField {}
pub trait FNField<const N: usize>: FMultiField {}

impl<const N: usize> CNField<N> for CMatrix<N> {}
impl<const N: usize> FNField<N> for FMatrix<N> {}
