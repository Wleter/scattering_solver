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

    fn value(&self, r: &f64) -> Self::Space {
        self.potentials.iter().fold(Self::Space::zero(), |acc, p| acc + p.value(r))
    }

    fn size(&self) -> usize {
        self.potentials.first().unwrap().size()
    }
}
