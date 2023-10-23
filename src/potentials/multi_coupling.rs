use crate::types::FMatrix;

use super::potential::Potential;

/// Multi coupling potential used to couple multi channel potentials.
#[derive(Clone)]
pub struct MultiCoupling<const N: usize, P>
where
    P: Potential<Space = f64>,
{
    potentials: Vec<(P, usize, usize)>,
    value_array: FMatrix<N>,

    symmetric: bool,
}

impl<const N: usize, P> MultiCoupling<N, P>
where
    P: Potential<Space = f64>,
{
    /// Creates new multi coupling potential with given vector of potentials with their coupling indices in potential matrix.
    /// If `symmetric` is true, the coupling matrix will be symmetric.
    pub fn new(potentials: Vec<(P, usize, usize)>, symmetric: bool) -> Self {
        Self {
            potentials,
            value_array: FMatrix::zeros(),
            symmetric,
        }
    }
}

impl<const N: usize, P> Potential for MultiCoupling<N, P>
where
    P: Potential<Space = f64>,
{
    type Space = FMatrix<N>;

    #[inline(always)]
    fn value(&mut self, r: &f64) -> Self::Space {
        for (potential, i, j) in self.potentials.iter_mut() {
            self.value_array[(*i, *j)] = potential.value(r);

            if self.symmetric {
                self.value_array[(*j, *i)] = self.value_array[(*i, *j)];
            }
        }

        self.value_array
    }
}
