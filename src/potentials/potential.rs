use num_traits::{One, Zero};
use std::iter::Sum;

/// Trait defining potential functionality
pub trait Potential: Clone {
    type Space: Sum + Zero + One;

    fn value(&mut self, r: &f64) -> Self::Space;
}