use crate::types::FMatrix;

use super::potential::Potential;

pub struct MultiChanPotential<const N: usize, P: Potential> {
    potentials: [P; N],
    value_array: FMatrix<N>,
}

impl<const N: usize, P> Potential for MultiChanPotential<N, P>
where
    P: Potential<Space = f64>,
{
    type Space = FMatrix<N>;

    fn value(&mut self, r: &f64) -> Self::Space {
        for (i, potential) in self.potentials.iter_mut().enumerate() {
            self.value_array[(i, i)] = potential.value(r);
        }

        self.value_array
    }
}
