use std::iter::Sum;

use num_traits::Zero;

/// Trait defining potential functionality
pub trait Potential
{
    type T: Sum + Zero;
    fn value(&mut self, r: f64) -> Self::T;
}