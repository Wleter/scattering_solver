use std::iter::Sum;

use num_traits::{One, Zero};

use super::potential::Potential;

/// Potential that gives `value` according to provided function.
#[derive(Clone)]
pub struct FunctionPotential<T: Clone, F: Fn(&f64) -> T> {
    function: F,
}

impl<T: Clone, F: Fn(&f64) -> T> FunctionPotential<T, F> {
    /// Creates new function potential with given function.
    pub fn new(function: F) -> Self {
        Self { function }
    }
}

impl<T, F> Potential for FunctionPotential<T, F>
where
    T: Clone + One + Zero + Sum,
    F: Fn(&f64) -> T + Clone + Send + Sync,
{
    type Space = T;

    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut Self::Space) {
        *destination = (self.function)(r)
    }
}
