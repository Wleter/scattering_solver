use crate::types::FMatrix;

use super::potential::Potential;

pub struct MultiChanPotential<const N: usize, P: Potential> {
    potentials: [P; N],
    value_array: FMatrix<N>,
}

impl<const N: usize, P: Potential<T = f64>> Potential for MultiChanPotential<N, P> {
    type T = FMatrix<N>;

    fn value(&mut self, r: f64) -> Self::T {
        for (i, potential) in self.potentials.iter_mut().enumerate() {
            self.value_array[(i, i)] = potential.value(r);
        }

        self.value_array
    }
}