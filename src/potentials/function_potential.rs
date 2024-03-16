use crate::types::{DFMatrix, FMatrix};

use super::potential::Potential;

/// Potential that gives `value` according to provided function.
#[derive(Clone)]
pub struct FunctionPotential<T, F: Fn(&f64) -> T + Send + Sync> {
    function: F,
}

impl<T: Clone, F: Fn(&f64) -> T + Send + Sync> FunctionPotential<T, F> {
    /// Creates new function potential with given function.
    pub fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F> Potential for FunctionPotential<f64, F>
where
    F: Fn(&f64) -> f64 + Send + Sync + Clone
{
    type Space = f64;

    fn value(&self, r: &f64) -> Self::Space {
        (self.function)(r)
    }

    fn size(&self) -> usize {
        1
    }
}

impl<const N: usize, F> Potential for FunctionPotential<FMatrix<N>, F>
where
    F: Fn(&f64) -> FMatrix<N> + Send + Sync + Clone
{
    type Space = FMatrix<N>;

    fn value(&self, r: &f64) -> Self::Space {
        (self.function)(r)
    }

    fn size(&self) -> usize {
        N
    }
}

impl<F> Potential for FunctionPotential<DFMatrix, F>
where
    F: Fn(&f64) -> DFMatrix + Send + Sync + Clone
{
    type Space = DFMatrix;

    fn value(&self, r: &f64) -> Self::Space {
        (self.function)(r)
    }

    fn size(&self) -> usize {
        Self::asymptotic_value(self).nrows()
    }
}