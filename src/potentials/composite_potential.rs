use super::potential::PotentialCurve;

/// Composite potential that has value equal to the sum of its components
#[derive(Debug, Clone)]
pub struct CompositePotential<P: PotentialCurve> {
    potentials: Vec<P>,
}

impl<P: PotentialCurve> CompositePotential<P> {
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

impl<P: PotentialCurve> PotentialCurve for CompositePotential<P>
{
    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut f64) {
        *destination = self.potentials.iter().fold(0.0, |acc, p| acc + p.value(r))
    }
}
