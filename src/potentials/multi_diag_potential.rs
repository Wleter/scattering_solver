use nalgebra::{Const, Dim, Dyn};

use crate::types::{DFMatrix, FMatrix};

use super::potential::Potential;

#[derive(Debug, Clone)]
pub struct MultiDiagPotential<N: Dim, P: Potential> {
    potentials: Vec<P>,
    size: N,
}

impl<const N: usize, P: Potential + Clone> MultiDiagPotential<Const<N>, P> {
    pub fn new(potentials: [P; N]) -> Self {
        Self { 
            potentials: potentials.to_vec(),
            size: Const::<N>,
        }
    }
}

impl<P: Potential> MultiDiagPotential<Dyn, P> {
    pub fn from_vec(potentials: Vec<P>) -> Self {
        let n = potentials.len();
        Self { 
            potentials,
            size: Dyn(n),
        }
    }
}

impl<const N: usize, P> Potential for MultiDiagPotential<Const<N>, P>
where
    P: Potential<Space = f64>,
{
    type Space = FMatrix<N>;

    fn value(&self, r: &f64) -> FMatrix<N> {
        let mut result = FMatrix::<N>::zeros();

        for (i, potential) in self.potentials.iter().enumerate() {
            result[(i, i)] = potential.value(r);
        }

        result
    }

    fn size(&self) -> usize {
        N
    }
}

impl<P> Potential for MultiDiagPotential<Dyn, P>
where
    P: Potential<Space = f64>,
{
    type Space = DFMatrix;

    fn value(&self, r: &f64) -> DFMatrix {
        let mut result = DFMatrix::zeros(self.size.0, self.size.0);

        for (i, potential) in self.potentials.iter().enumerate() {
            result[(i, i)] = potential.value(r);
        }

        result
    }

    fn size(&self) -> usize {
        self.size.0
    }
}