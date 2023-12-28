use std::iter::Sum;

use num_traits::Zero;

use super::potential::Potential;

/// Composite potential that has value equal to the sum of its components
#[derive(Debug, Clone)]
pub struct CompositePotential<P: Potential> {
    potentials: Vec<P>,
}

impl<P: Potential> CompositePotential<P> {
    pub fn new() -> Self {
        Self {
            potentials: Vec::new(),
        }
    }

    pub fn add_potential(&mut self, potential: P) -> &mut Self {
        self.potentials.push(potential);

        self
    }
}

impl<P: Potential> Potential for CompositePotential<P> 
where
    P::Space: Zero + Sum,
{
    type Space = P::Space;

    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut Self::Space) {
        *destination = self.potentials.iter().fold(Self::Space::zero(), |acc, p| acc + p.value(r))
    }
}
