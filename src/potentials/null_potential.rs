use super::potential::Potential;

/// Potential that is 0 everywhere
#[derive(Debug, Clone)]
pub struct NullPotential;

impl Potential for NullPotential {
    type Space = f64;

    fn value(&self, _r: &f64) -> f64 {
        0.0
    }

    fn size(&self) -> usize {
        1
    }
}
