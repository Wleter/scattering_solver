use super::potential::OnePotential;

/// Potential that gives `value` according to provided function.
#[derive(Clone)]
pub struct FunctionPotential<F: Fn(&f64) -> f64> {
    function: F,
}

impl<F: Fn(&f64) -> f64> FunctionPotential<F> {
    /// Creates new function potential with given function.
    pub fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F> OnePotential for FunctionPotential<F>
where
    F: Fn(&f64) -> f64 + Clone,
{
    #[inline(always)]
    fn value(&self, r: &f64) -> f64 {
        (self.function)(r)
    }
}
