use crate::types::FMultiField;

use super::potential::{PotentialSurface, PotentialCurve};

#[derive(Debug, Clone)]
pub struct MultiDiagPotential<const N: usize, P: PotentialCurve> {
    potentials: [P; N],
}

impl<const N: usize, P: PotentialCurve> MultiDiagPotential<N, P> {
    pub fn new(potentials: [P; N]) -> Self {
        Self { potentials }
    }
}

impl<const N: usize, P, T> PotentialSurface<T> for MultiDiagPotential<N, P>
where
    P: PotentialCurve,
    T: FMultiField
{
    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut T) {
        for (i, potential) in self.potentials.iter().enumerate() {
            destination[(i, i)] = potential.value(r);
        }
    }
}