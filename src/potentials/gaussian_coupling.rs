use quantum::units::{Unit, energy_units::Energy};

use super::potential::PotentialCurve;

/// Gaussian coupling potential
#[derive(Clone)]
pub struct GaussianCoupling {
    width: f64,
    center: f64,
    strength: f64,
}

impl GaussianCoupling {
    /// Creates new Gaussian coupling potential with given strength, center position and width
    pub fn new<U: Unit>(strength: Energy<U>, center: f64, width: f64) -> Self {
        Self {
            width,
            center,
            strength: strength.to_au(),
        }
    }
}

impl PotentialCurve for GaussianCoupling {
    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut f64) {
        *destination = self.strength * (-((r - self.center) / self.width).powi(2) / 2.0).exp()
    }
}
