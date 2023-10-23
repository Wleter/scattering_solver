use super::potential::Potential;

/// Potential of the form d0 * r^n + v0
#[derive(Debug, Clone)]
pub struct DispersionPotential {
    d0: f64,
    n: i32,
    v0: f64,
}

impl DispersionPotential {
    pub fn new(d0: f64, n: i32, v0: f64) -> Self {
        Self { d0, n, v0 }
    }
}

impl Potential for DispersionPotential {
    type Space = f64;

    #[inline(always)]
    fn value(&mut self, r: &f64) -> f64 {
        self.d0 * r.powi(self.n) + self.v0
    }
}
