use crate::types::{DFMatrix, FMatrix};

use super::potential::Potential;

#[derive(Clone)]
pub struct CoupledPotential<T, P, C>
where
    P: Potential<Space = T>,
    C: Potential<Space = T>,
{
    potential: P,
    coupling: C,
}

impl<T, P, C> CoupledPotential<T, P, C>
where
    P: Potential<Space = T>,
    C: Potential<Space = T>,
{
    pub fn new(potential: P, coupling: C) -> Self {
        Self {
            potential,
            coupling,
        }
    }
}

impl<const N: usize, P, C> Potential for CoupledPotential<FMatrix<N>, P, C>
where
    P: Potential<Space = FMatrix<N>>,
    C: Potential<Space = FMatrix<N>>,
{
    type Space = FMatrix<N>;

    fn value(&self, r: &f64) -> FMatrix<N> {
        self.potential.value(r) + self.coupling.value(r)
    }

    fn size(&self) -> usize {
        N
    }
}

impl<P, C> Potential for CoupledPotential<DFMatrix, P, C>
where
    P: Potential<Space = DFMatrix>,
    C: Potential<Space = DFMatrix>,
{
    type Space = DFMatrix;

    fn value(&self, r: &f64) -> DFMatrix {
        self.potential.value(r) + self.coupling.value(r)
    }

    fn size(&self) -> usize {
        self.potential.size()
    }
}