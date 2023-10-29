use crate::types::FMatrix;

use super::potential::Potential;

#[derive(Clone)]
pub struct CoupledPotential<const N: usize, P, C>
where
    P: Potential<Space = FMatrix<N>>,
    C: Potential<Space = FMatrix<N>>,
{
    potential: P,
    coupling: C,
}

impl<const N: usize, P, C> CoupledPotential<N, P, C>
where
    P: Potential<Space = FMatrix<N>>,
    C: Potential<Space = FMatrix<N>>,
{
    pub fn new(potential: P, coupling: C) -> Self {
        Self {
            potential,
            coupling,
        }
    }
}

impl<const N: usize, P, C> Potential for CoupledPotential<N, P, C>
where
    P: Potential<Space = FMatrix<N>>,
    C: Potential<Space = FMatrix<N>>,
{
    type Space = FMatrix<N>;

    #[inline(always)]
    fn value(&self, r: &f64) -> Self::Space {
        self.potential.value(r) + self.coupling.value(r)
    }
}
