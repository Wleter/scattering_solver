use super::potential::Potential;

/// Potential that is 0 everywhere
#[derive(Debug, Clone)]
pub struct NullPotential;

impl Potential for NullPotential {
    type Space = f64;

    fn value_inplace(&self, _r: &f64, destination: &mut f64) {
        *destination = 0.0;
    }
}
