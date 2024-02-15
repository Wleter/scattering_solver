use super::potential::PotentialCurve;

/// Potential that gives `value` according to provided function.
#[derive(Clone)]
pub struct FunctionPotential<F: Fn(&f64) -> f64 + Sync + Send + Clone>{
    function: F,
}

impl<F: Fn(&f64) -> f64 + Sync + Send + Clone> FunctionPotential<F> {
    /// Creates new function potential with given function.
    pub fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F: Fn(&f64) -> f64 + Sync + Send + Clone> PotentialCurve for FunctionPotential<F>
{
    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut f64) {
        *destination = (self.function)(r)
    }
}