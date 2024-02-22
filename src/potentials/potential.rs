use num_traits::{One, Zero};
use std::iter::Sum;

/// Trait defining potential functionality
pub trait Potential: Clone + Send + Sync {
    type Space: Sum + Zero + One;

    fn value_inplace(&self, r: &f64, destination: &mut Self::Space);

    fn value(&self, r: &f64) -> Self::Space {
        let mut destination = Self::Space::zero();
        self.value_inplace(r, &mut destination);
        destination
    }

    fn asymptotic_value(&self) -> Self::Space {
        self.value(&f64::INFINITY)
    }
}
