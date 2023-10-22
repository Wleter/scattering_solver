use super::potential::Potential;

/// Composite potential that has value equal to the sum of its components
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

impl<P: Potential> Potential for CompositePotential<P> {
    type Space = P::Space;

    fn value(&mut self, r: &f64) -> Self::Space {
        self.potentials.iter_mut().map(|p| p.value(r)).sum()
    }
}
