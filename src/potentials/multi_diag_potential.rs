use crate::types::FMatrix;

use super::potential::Potential;

#[derive(Debug, Clone)]
pub struct MultiDiagPotential<const N: usize, P: Potential> {
    potentials: [P; N],
}

impl<const N: usize, P: Potential> MultiDiagPotential<N, P> {
    pub fn new(potentials: [P; N]) -> Self {
        Self { potentials }
    }
}

impl<const N: usize, P> Potential for MultiDiagPotential<N, P>
where
    P: Potential<Space = f64>,
{
    type Space = FMatrix<N>;

    #[inline(always)]
    fn value(&self, r: &f64) -> Self::Space {
        let mut value_array = FMatrix::<N>::zeros();
        for (i, potential) in self.potentials.iter().enumerate() {
            value_array[(i, i)] = potential.value(r);
        }

        value_array
    }
}
