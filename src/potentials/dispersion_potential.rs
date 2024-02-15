use quantum::units::{distance_units::Distance, energy_units::Energy, Unit};

use super::potential::PotentialCurve;

/// Potential of the form d0 * r^n + v0
#[derive(Debug, Clone)]
pub struct DispersionPotential {
    d0: f64,
    n: i32,
}

impl DispersionPotential {
    pub fn new<V: Unit, U: Unit>(d0: f64, distance_unit: V, energy_unit: U, n: i32) -> Self {
        Self { 
            d0: d0 * Energy(1.0, energy_unit).to_au() * Distance(1.0, distance_unit).to_au().powi(-n), 
            n
        }
    }
}

impl PotentialCurve for DispersionPotential {
    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut f64) {
        *destination = self.d0 * r.powi(self.n)
    }
}
