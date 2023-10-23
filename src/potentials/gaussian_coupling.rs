use super::potential::Potential;

/// Gaussian coupling potential
#[derive(Clone)]
pub struct GaussianCoupling {
    width: f64,
    center: f64,
    strength: f64,
}

impl GaussianCoupling {
    /// Creates new Gaussian coupling potential with given strength, center position and width
    pub fn new(strength: f64, center: f64, width: f64) -> Self {
        Self {
            width,
            center,
            strength,
        }
    }
}

impl Potential for GaussianCoupling {
    type Space = f64;

    #[inline(always)]
    fn value(&mut self, r: &f64) -> f64 {
        self.strength * (-((r - self.center) / self.width).powi(2) / 2.0).exp()
    }
}
