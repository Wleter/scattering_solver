use crate::types::FMatrix;

use super::potential::Potential;

#[derive(Debug, Clone)]
pub struct MultiDiagPotential<const N: usize, P: Potential> {
    potentials: [P; N],
    value_array: FMatrix<N>,
}

impl<const N: usize, P: Potential> MultiDiagPotential<N, P> {
    pub fn new(potentials: [P; N]) -> Self {
        Self {
            potentials,
            value_array: FMatrix::zeros(),
        }
    }
}

impl<const N: usize, P> Potential for MultiDiagPotential<N, P>
where
    P: Potential<Space = f64>,
{
    type Space = FMatrix<N>;

    #[inline(always)]
    fn value(&mut self, r: &f64) -> Self::Space {
        for (i, potential) in self.potentials.iter_mut().enumerate() {
            self.value_array[(i, i)] = potential.value(r);
        }

        self.value_array
    }
}
