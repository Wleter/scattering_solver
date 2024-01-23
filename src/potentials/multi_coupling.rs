use crate::types::FMatrix;

use super::potential::Potential;

/// Multi coupling potential used to couple multi channel potentials.
#[derive(Clone)]
pub struct MultiCoupling<const N: usize, P>
where
    P: Potential<Space = f64>,
{
    potentials: Vec<(P, usize, usize)>,

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
    fn value_inplace(&self, r: &f64, destination: &mut FMatrix<N>) {
        for (potential, i, j) in self.potentials.iter() {
            let value = potential.value(r);

            destination[(*i, *j)] = value;

            if self.symmetric {
                destination[(*j, *i)] = value;
            }
        }
    }
}
