use quantum::units::{Unit, energy_units::Energy};

use super::potential::Potential;

/// Potential of the form d0 * r^n + v0
#[derive(Debug, Clone)]
pub struct DispersionPotential {
    d0: f64,
    n: i32,
    v0: f64,
}

impl DispersionPotential {
    pub fn new<U: Unit>(d0: Energy<U>, n: i32, v0: f64) -> Self {
        Self { d0: d0.to_au(), n, v0 }
    }
}

impl Potential for DispersionPotential {
    type Space = f64;

    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut f64) {
        *destination = self.d0 * r.powi(self.n) + self.v0
    }
}
